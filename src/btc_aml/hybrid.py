"""Hybrid models: GNN output + original features -> tree models.

    python -m btc_aml.hybrid --config configs/quick.yaml

Idea: a GNN's hidden layer is a learned summary of each transaction's
neighbourhood. Tree models are the strongest classifiers on this data but
cannot look at the graph. So we hand the trees the GNN's summary as extra
columns. Two variants are trained:

  "emb"  (the classic recipe, as in the original paper)
         table row = [ original features | GNN embedding of the same node ]

  "oof"  (out-of-fold score; added to fix a flaw of the classic recipe)
         table row = [ original features | GNN illicit probability ]
         where, for TRAINING rows, the probability comes from a GNN that was
         trained WITHOUT that row's time block.

Why the second variant exists: in "emb", the embeddings of the training rows
come from a GNN that was itself trained on those rows' labels, so they are
unrealistically informative there. The tree learns to trust them, and is
then let down on new data, where the embeddings are less clean. "oof"
removes the mismatch the standard way (as in stacking): the training steps
are cut into K consecutive time blocks; each block is scored by a GNN
trained on the other blocks. Every training row then gets a GNN output of
the same honest, out-of-sample quality as a test row. We stack the
probability rather than the embedding because embeddings of K different GNNs
do not share a coordinate system (hidden unit 7 of one network means
something different in another), while a probability means the same thing
for all of them.

Leakage rule for both variants: validation/test rows always get their GNN
output from the checkpoints saved by train_gnn.py, which follow the protocol:

    selection phase   GNN trained on steps 1-29  -> used for val rows
    final phase       GNN trained on steps 1-34  -> used for test rows

So no GNN involved has ever seen a label from the period it is evaluated on.
"""

from __future__ import annotations

import numpy as np

from .data import load_bundle
from .models import build_model
from .neural import fit, node_embeddings, predict_scores
from .protocol import run_protocol
from .splits import build_graph_set, build_table_set, make_graph, step_mask
from .train_baselines import TableLearner, table_refs
from .train_gnn import best_gnn, checkpoint_path, load_checkpoint
from .utils import config_arg_parser, get_device, get_logger, load_config, set_seed

log = get_logger()


class HybridLearner:
    """A TableLearner whose tables are widened with GNN output.

    The GNN output depends on the seed (each seed has its own GNN), so the
    tables are built per seed and cached.
    """

    def __init__(self, cfg: dict, tree: str, features: str, gnn_kind: str, mode: str,
                 bundle: dict, tables: dict, graphs: dict, cache: dict | None = None):
        self.cfg = cfg
        self.tree = tree
        self.features = features
        self.gnn_kind = gnn_kind
        self.mode = mode          # "emb" or "oof"
        self.bundle = bundle
        self.tables = tables      # plain feature tables (see splits.build_table_set)
        self.graphs = graphs      # the matching graphs, to run the GNN on
        self.direction = cfg["gnn"]["direction"]
        self.device = get_device(cfg)
        # seed -> {table name -> extra columns}. Passing the same dict to the RF
        # and LightGBM learners lets them share the (expensive) GNN outputs.
        self._cache: dict[int, dict] = {} if cache is None else cache

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _rows(graph, idx: np.ndarray) -> np.ndarray:
        """Position inside `graph` of each global node index in `idx`."""
        node_idx = graph.node_idx.numpy()      # sorted ascending
        rows = np.searchsorted(node_idx, idx)
        assert (node_idx[rows] == idx).all()
        return rows

    def _out_of_fold_scores(self, ckpt: dict, fit_name: str, scaler, seed: int) -> np.ndarray:
        """Illicit probability for every node of graphs[fit_name], each produced
        by a GNN that did not train on that node's time block."""
        graph = self.graphs[fit_name]
        first, last = (self.cfg["split"]["train"][0],
                       self.cfg["split"]["train" if fit_name == "train" else "val"][1])
        blocks = np.array_split(np.arange(first, last + 1), self.cfg["hybrid"]["oof_blocks"])
        t_all = self.bundle["time_step"]
        whole = step_mask(t_all, first, last)
        scores = np.full(graph.num_nodes, np.nan, dtype=np.float32)
        for block in blocks:
            held_out = step_mask(t_all, int(block[0]), int(block[-1]))
            fit_graph = make_graph(self.bundle, whole & ~held_out, self.features, scaler, self.direction)
            block_graph = make_graph(self.bundle, held_out, self.features, scaler, self.direction)
            set_seed(seed)
            model = build_model(self.gnn_kind, ckpt["in_dim"], ckpt["hp"], self.direction)
            # Same recipe as the main GNN: its hyperparameters and epoch count.
            fit(model, fit_graph, lr=ckpt["hp"]["lr"], weight_decay=ckpt["hp"]["weight_decay"],
                max_epochs=ckpt["n_iter"], device=self.device)
            scores[self._rows(graph, block_graph.node_idx.numpy())] = predict_scores(
                model, block_graph, self.device
            )
        assert not np.isnan(scores).any()
        return scores

    def _extra_columns(self, seed: int) -> dict:
        """GNN output for every node of each of the four graphs (cached per seed)."""
        if seed in self._cache:
            return self._cache[seed]
        extra = {}
        # (checkpoint phase, table the trees are fitted on, table they are evaluated on)
        for phase, fit_name, eval_name in (("select", "train", "val"), ("final", "trainval", "test")):
            path = checkpoint_path(self.cfg, "main", self.gnn_kind, self.features, self.direction, seed, phase)
            model, ckpt = load_checkpoint(path)
            model.to(self.device)
            scaler = self.graphs[f"{phase}_scaler"]
            for name in (fit_name, eval_name):
                graph = self.graphs[name]
                if self.mode == "emb":
                    extra[name] = node_embeddings(model, graph, self.device)
                elif name == eval_name:
                    extra[name] = predict_scores(model, graph, self.device)[:, None]
                else:
                    extra[name] = self._out_of_fold_scores(ckpt, fit_name, scaler, seed)[:, None]
        self._cache[seed] = extra
        return extra

    def _tables_for(self, seed: int) -> dict:
        """The plain tables with the GNN columns appended on the right."""
        extra = self._extra_columns(seed)
        wide = {}
        for name, (X, y, t, idx) in self.tables.items():
            rows = self._rows(self.graphs[name], idx)
            wide[name] = (np.hstack([X, extra[name][rows]]), y, t, idx)
        return wide

    # ---- protocol interface ----------------------------------------------
    def fit_select(self, hp: dict, seed: int):
        return TableLearner(self.cfg, self.tree, self._tables_for(seed)).fit_select(hp, seed)

    def fit_final(self, hp: dict, seed: int, info: dict):
        return TableLearner(self.cfg, self.tree, self._tables_for(seed)).fit_final(hp, seed, info)


def run(cfg: dict) -> None:
    bundle = load_bundle(cfg)
    section = cfg["hybrid"]
    for features in cfg["feature_sets"]:
        kind = best_gnn(cfg, features) if section["gnn"] == "best" else section["gnn"]
        tables = build_table_set(bundle, cfg, features)
        graphs = build_graph_set(bundle, cfg, features, cfg["gnn"]["direction"])
        val_ref, test_ref = table_refs(tables)
        for mode in section["modes"]:
            cache: dict = {}  # GNN outputs, shared by the tree models of this mode
            for tree in section["trees"]:
                log.info("Hybrid %s on %s + %s %s", tree, features, kind, mode)
                settings = dict(cfg["baselines"][tree])
                grid = settings.pop("grid", None)
                learner = HybridLearner(cfg, tree, features, kind, mode, bundle, tables, graphs, cache)
                run_protocol(
                    cfg, learner, experiment="hybrid", model=f"{tree}+{mode}", features=features,
                    variant=kind, base_hp=settings, grid=grid, val_ref=val_ref, test_ref=test_ref,
                )


def main() -> None:
    args = config_arg_parser("Train tree models on features + GNN output.").parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
