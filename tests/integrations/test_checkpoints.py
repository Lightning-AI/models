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
@mock.patch("litmodels.io.cloud.sdk_upload_model")
@mock.patch("litmodels.integrations.checkpoints.Auth")
def test_lightning_checkpoint_callback(mock_auth, mock_upload_model, importing, tmp_path):
    if importing == "lightning":
        from lightning import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint
        from lightning.pytorch.demos.boring_classes import BoringModel
        from litmodels.integrations.checkpoints import LightningModelCheckpoint as LitModelCheckpoint
    elif importing == "pytorch_lightning":
        from litmodels.integrations.checkpoints import PTLightningModelCheckpoint as LitModelCheckpoint
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.demos.boring_classes import BoringModel

    # Validate inheritance
    assert issubclass(LitModelCheckpoint, ModelCheckpoint)

    mock_upload_model.return_value.name = "org-name/teamspace/model-name"

    trainer = Trainer(
        max_epochs=2,
        callbacks=LitModelCheckpoint(model_name="org-name/teamspace/model-name"),
    )
    trainer.fit(BoringModel())

    # expected_path = model_path % str(tmpdir) if "%" in model_path else model_path
    assert mock_upload_model.call_count == 2
    assert mock_upload_model.call_args_list == [
        mock.call(name="org-name/teamspace/model-name", path=mock.ANY, progress_bar=True, cloud_account=None),
        mock.call(name="org-name/teamspace/model-name", path=mock.ANY, progress_bar=True, cloud_account=None),
    ]

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
        from litmodels.integrations.checkpoints import PTLightningModelCheckpoint as LitModelCheckpoint

    ckpt = LitModelCheckpoint(model_name="org-name/teamspace/model-name")
    pickle.dumps(ckpt)
