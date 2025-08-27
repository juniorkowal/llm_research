import torch
import torch_npu

# pip install torch-tb-profiler-ascend


def cc():
    experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[
                            torch_npu.profiler.ExportType.Text,
                                    torch_npu.profiler.ExportType.Db
                                        ],
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                            record_op_args=True
                            )
    return experimental_config




def create_profiler(EAGER_MODE_PROF=True):

    experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                                                                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1, 
                                                                    record_op_args=EAGER_MODE_PROF )
    experimental_config=cc()

    prof = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=0, skip_first=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir"),
            experimental_config=experimental_config)

    return prof



a=torch.rand(3,111,256,512, device='npu')
b=torch.randint(low=1, high=250, size=[3,111,256,256],device='npu')
with create_profiler() as prof:
   for i in range(1,10):
     print(f'Executed {i}')
     c= torch.gather(a, dim=-1, index=b)
     prof.step()


print(prof)