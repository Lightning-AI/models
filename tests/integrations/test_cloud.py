import os
from contextlib import redirect_stdout
from io import StringIO

import pytest
from lightning_sdk import Teamspace
from lightning_sdk.lightning_cloud.rest_client import GridRestClient
from lightning_sdk.utils.resolve import _resolve_teamspace
from litmodels import download_model, upload_model

from tests.integrations import _SKIP_IF_LIGHTNING_BELLOW_2_5_1, _SKIP_IF_PYTORCHLIGHTNING_BELLOW_2_5_1

LIT_ORG = "lightning-ai"
LIT_TEAMSPACE = "LitModels"


def _prepare_variables(test_name: str) -> tuple[Teamspace, str, str]:
    model_name = f"litmodels_test_integrations_{test_name}+{os.urandom(8).hex()}"
    teamspace = _resolve_teamspace(org=LIT_ORG, teamspace=LIT_TEAMSPACE, user=None)
    org_team = f"{teamspace.owner.name}/{teamspace.name}"
    return teamspace, org_team, model_name


def _cleanup_model(teamspace: Teamspace, model_name: str) -> None:
    """Cleanup model from the teamspace."""
    client = GridRestClient()
    # cleaning created models as each test run shall have unique model name
    model = client.models_store_get_model_by_name(
        project_owner_name=teamspace.owner.name,
        project_name=teamspace.name,
        model_name=model_name,
    )
    client.models_store_delete_model(project_id=teamspace.id, model_id=model.id)


@pytest.mark.cloud()
def test_upload_download_model(tmp_path):
    """Verify that the model is uploaded to the teamspace"""
    # create a dummy file
    file_path = tmp_path / "dummy.txt"
    with open(file_path, "w") as f:
        f.write("dummy")

    # model name with random hash
    teamspace, org_team, model_name = _prepare_variables("upload_download")

    out = StringIO()
    with redirect_stdout(out):
        upload_model(name=f"{org_team}/{model_name}", model=file_path)

    # validate the output
    assert (
        f"Model uploaded successfully. Link to the model: 'https://lightning.ai/{org_team}/models/{model_name}'"
    ) in out.getvalue()

    os.remove(file_path)
    assert not os.path.isfile(file_path)

    model_files = download_model(name=f"{org_team}/{model_name}", download_dir=tmp_path)
    assert model_files == ["dummy.txt"]
    for file in model_files:
        assert os.path.isfile(os.path.join(tmp_path, file))

    # CLEANING
    _cleanup_model(teamspace, model_name)


@pytest.mark.parametrize(
    "importing",
    [
        pytest.param("lightning", marks=_SKIP_IF_LIGHTNING_BELLOW_2_5_1),
        pytest.param("pytorch_lightning", marks=_SKIP_IF_PYTORCHLIGHTNING_BELLOW_2_5_1),
    ],
)
@pytest.mark.cloud()
# todo: mock env variables as it would run in studio
def test_lightning_default_checkpointing(importing, tmp_path):
    if importing == "lightning":
        from lightning import Trainer
        from lightning.pytorch.demos.boring_classes import BoringModel
    elif importing == "pytorch_lightning":
        from pytorch_lightning import Trainer
        from pytorch_lightning.demos.boring_classes import BoringModel

    # model name with random hash
    teamspace, org_team, model_name = _prepare_variables("default_checkpoint")

    trainer = Trainer(
        max_epochs=2,
        default_root_dir=tmp_path,
        model_registry=f"{org_team}/{model_name}",
    )
    trainer.fit(BoringModel())

    # CLEANING
    _cleanup_model(teamspace, model_name)


@pytest.mark.parametrize("trainer_method", ["fit", "validate", "test", "predict"])
@pytest.mark.parametrize(
    "registry", ["registry", "registry:version:v1", "registry:<model>", "registry:<model>:version:v1"]
)
@pytest.mark.parametrize(
    "importing",
    [
        pytest.param("lightning", marks=_SKIP_IF_LIGHTNING_BELLOW_2_5_1),
        pytest.param("pytorch_lightning", marks=_SKIP_IF_PYTORCHLIGHTNING_BELLOW_2_5_1),
    ],
)
@pytest.mark.cloud()
# todo: mock env variables as it would run in studio
def test_lightning_resume(trainer_method, registry, importing, tmp_path):
    if importing == "lightning":
        from lightning import Trainer
        from lightning.pytorch.demos.boring_classes import BoringModel
    elif importing == "pytorch_lightning":
        from pytorch_lightning import Trainer
        from pytorch_lightning.demos.boring_classes import BoringModel

    trainer = Trainer(max_epochs=1, limit_train_batches=50, limit_val_batches=20, default_root_dir=tmp_path)
    trainer.fit(BoringModel())
    checkpoint_path = getattr(trainer.checkpoint_callback, "best_model_path")

    # model name with random hash
    teamspace, org_team, model_name = _prepare_variables("resume")
    upload_model(model=checkpoint_path, name=f"{org_team}/{model_name}")

    trainer_kwargs = {"model_registry": f"{org_team}/{model_name}"} if "<model>" not in registry else {}
    trainer = Trainer(
        max_epochs=2,
        default_root_dir=tmp_path,
        limit_train_batches=10,
        limit_val_batches=10,
        limit_test_batches=10,
        limit_predict_batches=10,
        **trainer_kwargs,
    )
    registry = registry.replace("<model>", f"{org_team}/{model_name}")
    if trainer_method == "fit":
        trainer.fit(BoringModel(), ckpt_path=registry)
    elif trainer_method == "validate":
        trainer.validate(BoringModel(), ckpt_path=registry)
    elif trainer_method == "test":
        trainer.test(BoringModel(), ckpt_path=registry)
    elif trainer_method == "predict":
        trainer.predict(BoringModel(), ckpt_path=registry)
    else:
        raise ValueError(f"Unknown trainer method: {trainer_method}")

    # CLEANING
    _cleanup_model(teamspace, model_name)
