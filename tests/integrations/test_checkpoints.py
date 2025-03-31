import pickle
import re
from unittest import mock

import pytest

from tests.integrations import _SKIP_IF_LIGHTNING_MISSING, _SKIP_IF_PYTORCHLIGHTNING_MISSING


@pytest.mark.parametrize(
    "importing",
    [
        pytest.param("lightning", marks=_SKIP_IF_LIGHTNING_MISSING),
        pytest.param("pytorch_lightning", marks=_SKIP_IF_PYTORCHLIGHTNING_MISSING),
    ],
)
@pytest.mark.parametrize("model_name", [None, "org-name/teamspace/model-name"])
@mock.patch("litmodels.integrations.checkpoints.LitModelCheckpointMixin._datetime_stamp", return_value="20250102-1213")
@mock.patch("litmodels.io.cloud.sdk_upload_model")
@mock.patch("litmodels.integrations.checkpoints.Auth")
def test_lightning_checkpoint_callback(
    mock_auth, mock_upload_model, mock_datetime_stamp, monkeypatch, importing, model_name, tmp_path
):
    if importing == "lightning":
        from lightning import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint
        from lightning.pytorch.demos.boring_classes import BoringModel
        from litmodels.integrations.checkpoints import LightningModelCheckpoint as LitModelCheckpoint
    elif importing == "pytorch_lightning":
        from litmodels.integrations.checkpoints import PytorchLightningModelCheckpoint as LitModelCheckpoint
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.demos.boring_classes import BoringModel

    # Validate inheritance
    assert issubclass(LitModelCheckpoint, ModelCheckpoint)

    ckpt_args = {"model_name": model_name} if model_name else {}
    expected_model_registry = ckpt_args.get("model_name", f"BoringModel_{LitModelCheckpoint._datetime_stamp}")
    mock_upload_model.return_value.name = expected_model_registry
    monkeypatch.setattr(
        "lightning_sdk.utils.resolve._resolve_teamspace",
        mock.MagicMock(return_value=mock.MagicMock(owner=mock.MagicMock(name="my-org"), name="dream-team")),
    )

    trainer = Trainer(
        max_epochs=2,
        callbacks=LitModelCheckpoint(**ckpt_args),
    )
    trainer.fit(BoringModel())

    assert mock_auth.call_count == 1
    assert mock_upload_model.call_args_list == [
        mock.call(name=expected_model_registry, path=mock.ANY, progress_bar=True, cloud_account=None),
        mock.call(name=expected_model_registry, path=mock.ANY, progress_bar=True, cloud_account=None),
    ]
    called_name_related_mocks = 1
    mock_datetime_stamp.call_count == called_name_related_mocks
    mock_resolve_teamspace.call_count == called_name_related_mocks

    # Verify paths match the expected pattern
    for call_args in mock_upload_model.call_args_list:
        path = call_args[1]["path"]
        assert re.match(r".*[/\\]lightning_logs[/\\]version_\d+[/\\]checkpoints[/\\]epoch=\d+-step=\d+\.ckpt$", path)


@pytest.mark.parametrize(
    "importing",
    [
        pytest.param("lightning", marks=_SKIP_IF_LIGHTNING_MISSING),
        pytest.param("pytorch_lightning", marks=_SKIP_IF_PYTORCHLIGHTNING_MISSING),
    ],
)
@mock.patch("litmodels.integrations.checkpoints.Auth")
def test_lightning_checkpointing_pickleable(mock_auth, importing):
    if importing == "lightning":
        from litmodels.integrations.checkpoints import LightningModelCheckpoint as LitModelCheckpoint
    elif importing == "pytorch_lightning":
        from litmodels.integrations.checkpoints import PytorchLightningModelCheckpoint as LitModelCheckpoint

    ckpt = LitModelCheckpoint(model_name="org-name/teamspace/model-name")
    pickle.dumps(ckpt)
