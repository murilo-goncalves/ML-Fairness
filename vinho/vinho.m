data = csvread("winequality-white.csv");

y = data(:,12);
m = length(y);
X = [ones(m, 1), data(:,1:11)];

theta = pinv(X'*X)*X'*y;

result = X*theta