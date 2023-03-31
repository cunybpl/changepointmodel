import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.abspath(os.path.join(ROOT_DIR, "tests", "fixtures"))

GENERATED_DATA_ALL_MODELS_FILE = os.path.join(
    FIXTURES_DIR, "generated_data_all_models.json"
)
GENERATED_DATA_POST = os.path.join(FIXTURES_DIR, "generated_data_for_post.json")
ENERGYMODEL_DATAPATH = os.path.join(FIXTURES_DIR, "energymodel-fixture.json")
