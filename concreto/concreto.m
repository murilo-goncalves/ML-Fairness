data = csvread("Concrete_Data.csv");

y = data(:,end);
m = length(y);
X = [ones(m, 1), data(:,1:8)];

theta = pinv(X'*X)*X'*y;

result = X*theta