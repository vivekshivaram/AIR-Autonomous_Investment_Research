import src.memory as memory
from src.bootstrap import build_runtime
from src.config import Settings
from tests.report_fakes import FakeLLM


def test_runtime_wires_the_report_pipeline_and_memory_tools(tmp_path):
    memory.use_database(tmp_path / "memory.db", reset=True)
    runtime = build_runtime(settings=Settings(output_dir=tmp_path), llm=FakeLLM(), allow_stubs=True)

    assert runtime.integration["report_pipeline"] == "real"
    assert runtime.integration["memory"] == "real"
    assert runtime.integration["llm_cache"] == "off"  # an explicitly passed client is used as given
    assert runtime.tools.has("memory_retrieval") and runtime.tools.has("memory_storage")
    assert runtime.orchestrator.report_pipeline is not None
