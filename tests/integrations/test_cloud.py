import os
from contextlib import redirect_stdout
from io import StringIO


import pytest
from lightning_sdk.lightning_cloud.rest_client import GridRestClient
from lightning_sdk.utils.resolve import _resolve_teamspace

from litmodels import upload_model, download_model

LIT_ORG = "lightning-ai"
LIT_TEAMSPACE = "LitModels"


@pytest.mark.cloud()
def test_upload_download_model(tmp_path):
    """Verify that the model is uploaded to the teamspace"""
    # create a dummy file
    file_path = tmp_path / "dummy.txt"
    with open(file_path, "w") as f:
        f.write("dummy")

    # model name with random hash
    model_name = f"litmodels_test_integrations+{os.urandom(8).hex()}"
    teamspace = _resolve_teamspace(org=LIT_ORG, teamspace=LIT_TEAMSPACE, user=None)
    org_team = f"{teamspace.owner.name}/{teamspace.name}"

    out = StringIO()
    with redirect_stdout(out):
        upload_model(
            name=f"{org_team}/{model_name}",
            model=file_path
        )

    # validate the output
    assert (
        "Model uploaded successfully. Link to the model:"
        f" 'https://lightning.ai/{org_team}/models/{model_name}'"
    ) in out.getvalue()

    os.remove(file_path)
    assert not os.path.isfile(file_path)

    model_files = download_model(name=f"{org_team}/{model_name}", download_dir=tmp_path)
    assert model_files, "Model files not downloaded"
    for file in model_files:
        assert os.path.isfile(os.path.join(tmp_path, file))

    client = GridRestClient()
    # cleaning created models with todo: also consider how to delete just this version of the model
    model = client.models_store_get_model_by_name(
        project_owner_name=teamspace.owner.name,
        project_name=teamspace.name,
        model_name=model_name,
    )
    client.models_store_delete_model(project_id=teamspace.id, model_id=model.id)