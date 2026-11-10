"""lookup status tool tests."""

from harness_lab.fake_tool import lookup_status


def test_fake_tool_return_company_status():
    """Verify lookup_status returns the status for a project."""
    project_name = "Atlas"
    status = lookup_status(project_name)
    assert status == "Project: " + project_name + " Status: OK"
