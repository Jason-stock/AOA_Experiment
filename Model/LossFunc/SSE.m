function [sse] = SSE(Y_output, Y_target)
div = Y_output-Y_target;
sse = (div'*div);
end