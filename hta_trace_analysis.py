from hta.trace_analysis import TraceAnalysis
import os
from hta.configs.parser_config import ParserConfig
from hta.configs.config import HtaConfig
from hta.common.trace import Trace


cwd = os.getcwd()

trace_dir = f"{cwd}/result_dir/ubuntu_131542_20250828141349066_ascend_pt"
# trace_dir = f"{cwd}/llm_research/result_dir/ubuntu_131542_20250828141349066_ascend_pt/ASCEND_PROFILER_OUTPUT/"
test_trace_dir = HtaConfig.get_test_data_path(trace_dir)

def check_trace(trace_dir: str):
    t = Trace(trace_dir=trace_dir)
    t.parse_traces(use_multiprocessing=False)
    rank = next(iter(t.traces))
    df = t.get_trace(rank)
    df.info()
check_trace(test_trace_dir)


analyzer = TraceAnalysis(trace_dir=trace_dir, trace_files=['trace_view.json']
                         )



# import pandas as pd
# # Load your trace data (replace 'trace_file.json' with your file)
# df = pd.read_json('/root/llm_research/result_dir/ubuntu_131542_20250828141349066_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json')
# print("Columns in trace data:", df.columns.tolist())