
def default_output_model_name(base_model: str, dataset: str):
    base_model_stem = base_model.split("/")[-1].strip("/")
    dataset_stem = dataset.split("/")[-1].strip("/")
    return f"{base_model_stem}_{dataset_stem}"