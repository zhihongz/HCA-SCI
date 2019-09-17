function y = piecewise_linear_tfrom(x, x_low, x_high, y_low, y_high)
% truncate x to limit it into [x_low, x_high] and then rescale it to [y_low,
% y_high], the y-x function's shape is like "_/-"
% input: 
%       x: input self-variable
%       [x_low, x_high]: the lower inflection point
%       [y_low, y_higt]: the higher inflection point
% output:
%       y: converted result
% 

% truncation
x = min(x, x_high);
x = max(x, x_low);

% linear transformation
y = (x-x_low)/(x_high-x_low)*(y_high-y_low)+y_low;
end