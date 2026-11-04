"""lookup status tool tests."""

from harness_lab.fake_tool import lookup_status


class TestFakeTool:
    """Tests for the fake status lookup tool."""

    def test_fake_tool_return_company_status(self):
        """Verify lookup_status returns the status for a project."""
        project_name = "Atlas"
        status = lookup_status(project_name)
        assert status == "Project: " + project_name + " Status: OK"
