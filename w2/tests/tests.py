import sys

import torch
import torch.nn.functional as F

return_string: str = ""

def func(code: str) -> str:
    global return_string
    return_string = ""
    exec(code, globals())
    return return_string

tests: list[str] = [
    """
    input = torch.arange(1,17,dtype=torch.float32).reshape(1,1,4,4);
    kernel = torch.tensor([[[[1.0]]]], dtype=torch.float32);
    output = F.conv2d(input,kernel,padding=0);
    vals_in = ','.join(f'{x:.6f}' for x in input.flatten().tolist());
    vals_k = ','.join(f'{x:.6f}' for x in kernel.flatten().tolist());
    vals_out = ','.join(f'{x:.6f}' for x in output.flatten().tolist());
    return_string = 'INPUT_SHAPE: 1,1,4,4$$INPUT_VALUES: ' + vals_in + '$$KERNEL_SHAPE: 1,1,1,1$$KERNEL_VALUES: ' + vals_k + '$$PADDING: 0$$STRIDE: 1,1$$DILATION: 1,1$$OUTPUT_SHAPE: ' + ','.join(map(str,output.shape)) + '$$OUTPUT_VALUES: ' + vals_out;
    """,
    """
    input = torch.arange(1,26,dtype=torch.float32).reshape(1,1,5,5);
    kernel = torch.arange(1,10,dtype=torch.float32).reshape(1,1,3,3);
    output = F.conv2d(input,kernel,padding=0);
    vals_in = ','.join(f'{x:.6f}' for x in input.flatten().tolist());
    vals_k = ','.join(f'{x:.6f}' for x in kernel.flatten().tolist());
    vals_out = ','.join(f'{x:.6f}' for x in output.flatten().tolist());
    return_string = 'INPUT_SHAPE: 1,1,5,5$$INPUT_VALUES: ' + vals_in + '$$KERNEL_SHAPE: 1,1,3,3$$KERNEL_VALUES: ' + vals_k + '$$PADDING: 0$$STRIDE: 1,1$$DILATION: 1,1$$OUTPUT_SHAPE: ' + ','.join(map(str,output.shape)) + '$$OUTPUT_VALUES: ' + vals_out;
    """
    ,"""
    input = torch.arange(1,37,dtype=torch.float32).reshape(1,1,6,6);
    kernel = torch.arange(1,10,dtype=torch.float32).reshape(1,1,3,3);
    output = F.conv2d(input,kernel,padding=0,stride=2,dilation=1);
    vals_in = ','.join(f'{x:.6f}' for x in input.flatten().tolist());
    vals_k = ','.join(f'{x:.6f}' for x in kernel.flatten().tolist());
    vals_out = ','.join(f'{x:.6f}' for x in output.flatten().tolist());
    return_string = 'INPUT_SHAPE: 1,1,6,6$$INPUT_VALUES: ' + vals_in + '$$KERNEL_SHAPE: 1,1,3,3$$KERNEL_VALUES: ' + vals_k + '$$PADDING: 0$$STRIDE: 2,2$$DILATION: 1,1$$OUTPUT_SHAPE: ' + ','.join(map(str,output.shape)) + '$$OUTPUT_VALUES: ' + vals_out;
    """,
    """
    input = torch.arange(1,26,dtype=torch.float32).reshape(1,1,5,5);
    kernel = torch.arange(1,10,dtype=torch.float32).reshape(1,1,3,3);
    # SAME padding -> pad=1 for kernel 3
    output = F.conv2d(input,kernel,padding=1,stride=1,dilation=1);
    vals_in = ','.join(f'{x:.6f}' for x in input.flatten().tolist());
    vals_k = ','.join(f'{x:.6f}' for x in kernel.flatten().tolist());
    vals_out = ','.join(f'{x:.6f}' for x in output.flatten().tolist());
    return_string = 'INPUT_SHAPE: 1,1,5,5$$INPUT_VALUES: ' + vals_in + '$$KERNEL_SHAPE: 1,1,3,3$$KERNEL_VALUES: ' + vals_k + '$$PADDING: SAME$$STRIDE: 1,1$$DILATION: 1,1$$OUTPUT_SHAPE: ' + ','.join(map(str,output.shape)) + '$$OUTPUT_VALUES: ' + vals_out;
    """,
    """
    input = torch.arange(1,50,dtype=torch.float32).reshape(1,1,7,7);
    kernel = torch.arange(1,10,dtype=torch.float32).reshape(1,1,3,3);
    output = F.conv2d(input,kernel,padding=0,stride=1,dilation=2);
    vals_in = ','.join(f'{x:.6f}' for x in input.flatten().tolist());
    vals_k = ','.join(f'{x:.6f}' for x in kernel.flatten().tolist());
    vals_out = ','.join(f'{x:.6f}' for x in output.flatten().tolist());
    return_string = 'INPUT_SHAPE: 1,1,7,7$$INPUT_VALUES: ' + vals_in + '$$KERNEL_SHAPE: 1,1,3,3$$KERNEL_VALUES: ' + vals_k + '$$PADDING: 0$$STRIDE: 1,1$$DILATION: 2,2$$OUTPUT_SHAPE: ' + ','.join(map(str,output.shape)) + '$$OUTPUT_VALUES: ' + vals_out;
    """,
    """
    input = torch.arange(1,1+1*2*4*4,dtype=torch.float32).reshape(1,2,4,4);
        outC, inC, kH, kW = 2, 2, 3, 3;
    kernel = torch.tensor([(i % 9) + 1 for i in range(outC*inC*kH*kW)], dtype=torch.float32).reshape(outC,inC,kH,kW);
    output = F.conv2d(input,kernel,padding=0,stride=1,dilation=1);
    vals_in = ','.join(f'{x:.6f}' for x in input.flatten().tolist());
    vals_k = ','.join(f'{x:.6f}' for x in kernel.flatten().tolist());
    vals_out = ','.join(f'{x:.6f}' for x in output.flatten().tolist());
    return_string = 'INPUT_SHAPE: 1,2,4,4$$INPUT_VALUES: ' + vals_in + '$$KERNEL_SHAPE: 2,2,3,3$$KERNEL_VALUES: ' + vals_k + '$$PADDING: 0$$STRIDE: 1,1$$DILATION: 1,1$$OUTPUT_SHAPE: ' + ','.join(map(str,output.shape)) + '$$OUTPUT_VALUES: ' + vals_out;
    """
]

def main():
    if len(sys.argv) != 2:
        raise ValueError("Invalid number of arguments")
    output_file = sys.argv[1]
    with open(output_file, "w") as f:
        for test in tests:
            # build a single-line code string while removing comment lines
            lines = [l.strip() for l in test.splitlines()]
            lines = [l for l in lines if l and not l.lstrip().startswith('#')]
            code = '\n'.join(lines).replace('$$', '\\n')
            f.write(func(code) + "\n")

if __name__ == "__main__":
    main()
