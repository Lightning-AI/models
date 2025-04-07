import os

import pytest
from lightning_sdk.lightning_cloud.rest_client import GridRestClient
from lightning_sdk.utils.resolve import _resolve_teamspace

from litmodels.integrations.duplicate import duplicate_hf_model
from tests.integrations import LIT_ORG, LIT_TEAMSPACE


@pytest.mark.cloud
def test_duplicate_hf_model(tmp_path):
    """Verify that the HF model can be duplicated to the teamspace"""

    # model name with random hash
    model_name = f"litmodels_hf_model+{os.urandom(8).hex()}"
    teamspace = _resolve_teamspace(org=LIT_ORG, teamspace=LIT_TEAMSPACE, user=None)
    org_team = f"{teamspace.owner.name}/{teamspace.name}"

    duplicate_hf_model(hf_model="google/t5-efficient-tiny", lit_model=f"{org_team}/{model_name}")

    client = GridRestClient()
    model = client.models_store_get_model_by_name(
        project_owner_name=teamspace.owner.name,
        project_name=teamspace.name,
        model_name=model_name,
    )
    client.models_store_delete_model(project_id=teamspace.id, model_id=model.id)
