from libcity.pipeline import run_model

def test_migration(model_name, dataset, task, paper_title, gpu, epochs=1):
    parameters = {
        'model': model_name,
        'dataset': dataset,
        'task': task,
        'gpu': gpu,
        'epochs': epochs,
        'config_file': f'libcity/config/model/{task}/{model_name}.json'
    }
    run_model(parameters)

test_migration(
    model_name="GraphMM",
    dataset="foursquare_tky",
    task="trajectory_loc_prediction",
    paper_title="GraphMM Migration Test",
    gpu="0",
    epochs=1
)
