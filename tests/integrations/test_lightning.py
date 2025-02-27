import re
from unittest import mock

from litmodels.integrations.imports import _LIGHTNING_AVAILABLE, _PYTORCHLIGHTNING_AVAILABLE
from litmodels.integrations.lightning_checkpoint import LitModelCheckpoint

if _LIGHTNING_AVAILABLE:
    from lightning import Trainer
    from lightning.pytorch.demos.boring_classes import BoringModel
elif _PYTORCHLIGHTNING_AVAILABLE:
    from pytorch_lightning import Trainer
    from pytorch_lightning.demos.boring_classes import BoringModel


@mock.patch("litmodels.io.cloud.sdk_upload_model")
def test_upload_model(mock_upload_model, tmp_path):
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
        assert re.match(r".*/lightning_logs/version_\d+/checkpoints/epoch=\d+-step=\d+\.ckpt$", path)
