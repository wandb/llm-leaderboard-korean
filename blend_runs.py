import wandb

api = wandb.Api()
entity = "wandb-korea"
project = "llm-leaderboard3"
source_run_id = "ntsryj53"
target_run_id = "1opb4yaw"


source_run = api.run(f"{entity}/{project}/{source_run_id}")
source_artifact_dir_list = []
for artifact in source_run.logged_artifacts():
    print(artifact.name)
    source_artifact_dir = artifact.download()
    source_artifact_dir_list.append(source_artifact_dir)

run = wandb.init(
    entity="wandb-korea",
    project="llm-leaderboard3",
    id=target_run_id,
    resume="allow",
)

for source_artifact_dir in source_artifact_dir_list:
    name = f'run-{target_run_id}-' + source_artifact_dir.split("/")[-1].split("-")[2].split(":")[0]
    artifact = wandb.Artifact(name=name, type="run_table")
    artifact.add_dir(source_artifact_dir)
    run.log_artifact(artifact)
run.finish()