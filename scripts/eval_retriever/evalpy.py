import os
import subprocess
from datetime import datetime

# Input arguments
import sys

def main():
    if len(sys.argv) < 9:
        print("Usage: python script.py <MAX_Q_LEN> <MAX_P_LEN> <PER_DEV_BATCH_SIZE> <GPUS_PER_NODE> <POOLING> <ATTENTION> <SUB_DATASET> <MODEL_PATH>")
        sys.exit(1)

    MAX_Q_LEN = int(sys.argv[1])
    MAX_P_LEN = int(sys.argv[2])
    PER_DEV_BATCH_SIZE = int(sys.argv[3])
    GPUS_PER_NODE = int(sys.argv[4])
    POOLING = sys.argv[5]
    ATTENTION = sys.argv[6]
    SUB_DATASET = sys.argv[7]
    MODEL_PATH = sys.argv[8]

    attn_implementation = 'sdpa' if 'SigLIP' not in MODEL_PATH else 'eager'

    WORLD_SIZE = 1
    RANK = 0
    MASTER_ENDPOINT = "localhost"
    MASTER_PORT = 23456

    CHECKPOINT_DIR = "/home/dzi/VisRAG.git/VisRAG/data/checkpoints"
    TIMESTR = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    IDENTITY = f"eval-{TIMESTR}-maxq-{MAX_Q_LEN}-maxp-{MAX_P_LEN}-bsz-{PER_DEV_BATCH_SIZE}-pooling-{POOLING}-attention-{ATTENTION}-gpus-per-node-{GPUS_PER_NODE}"
    RESULT_DIR = os.path.join(CHECKPOINT_DIR, IDENTITY)

    SUB_DATASET_LIST = SUB_DATASET.split(',')

    for sub_dataset in SUB_DATASET_LIST:
        print(f"Evaluating: {sub_dataset}")

        THIS_RESULT_DIR = os.path.join(RESULT_DIR, sub_dataset)
        print(f"This dataset result dir: {THIS_RESULT_DIR}")

        CORPUS_PATH = f"openbmb/VisRAG-Ret-Test-{sub_dataset}"
        QUERY_PATH = f"openbmb/VisRAG-Ret-Test-{sub_dataset}"
        QRELS_PATH = f"openbmb/VisRAG-Ret-Test-{sub_dataset}"

        print(f"CORPUS_PATH: {CORPUS_PATH}")
        print(f"QUERY_PATH: {QUERY_PATH}")
        print(f"QRELS_PATH: {QRELS_PATH}")

        QUERY_TEMPLATE = "Represent this query for retrieving relevant documents: <query>"
        CORPUS_TEMPLATE = "<text>"

        # Phase: encode
        subprocess.run([
            "torchrun",
            f"--nnodes={WORLD_SIZE}",
            f"--node_rank={RANK}",
            f"--nproc_per_node={GPUS_PER_NODE}",
            f"--master_addr={MASTER_ENDPOINT}",
            f"--master_port={MASTER_PORT}",
            "src/openmatch/driver/eval.py",
            f"--qrels_path={QRELS_PATH}",
            f"--query_path={QUERY_PATH}",
            f"--corpus_path={CORPUS_PATH}",
            f"--model_name_or_path={MODEL_PATH}",
            f"--output_dir={THIS_RESULT_DIR}",
            f"--query_template={QUERY_TEMPLATE}",
            f"--doc_template={CORPUS_TEMPLATE}",
            f"--q_max_len={MAX_Q_LEN}",
            f"--p_max_len={MAX_P_LEN}",
            f"--per_device_eval_batch_size={PER_DEV_BATCH_SIZE}",
            "--dataloader_num_workers=1",
            "--dtype=float16",
            "--use_gpu",
            "--overwrite_output_dir=false",
            "--max_inmem_docs=1000000",
            "--normalize=true",
            f"--pooling={POOLING}",
            f"--attention={ATTENTION}",
            f"--attn_implementation={attn_implementation}",
            "--phase=encode",
            "--from_hf_repo",
        ], check=True)

        # Phase: retrieve
        subprocess.run([
            "torchrun",
            f"--nnodes={WORLD_SIZE}",
            f"--node_rank={RANK}",
            f"--nproc_per_node={GPUS_PER_NODE}",
            f"--master_addr={MASTER_ENDPOINT}",
            f"--master_port={MASTER_PORT}",
            "src/openmatch/driver/eval.py",
            f"--model_name_or_path={MODEL_PATH}",
            f"--qrels_path={QRELS_PATH}",
            f"--query_path={QUERY_PATH}",
            f"--corpus_path={CORPUS_PATH}",
            f"--output_dir={THIS_RESULT_DIR}",
            "--use_gpu",
            "--phase=retrieve",
            "--retrieve_depth=10",
            "--from_hf_repo",
        ], check=True)

if __name__ == "__main__":
    main()
