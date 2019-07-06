%Gaussian Processes
%Examples.


% Choose a kernel (covariance function) 
kernel = 3; 
L = 100; %scaling
switch kernel
    case 1; k =@(x,y) 1*x'*y; % Linear 
    case 2; k =@(x,y) 1*min(x,y); % Brownian motion
    case 3; k =@(x,y) exp(-((x-y)'*(x-y)/(L^2))); % Squared exponential
    case 4; k =@(x,y) exp(-1*sqrt((x-y)'*(x-y))); % Ornstein-Uhlenbeck
    case 5; k =@(x,y) exp(-1*sin(50*pi*(x-y))^2); % Periodic
    case 6; k =@(x,y) exp(-100*min(abs(x-y), abs(x+y))^2); %symmetric ? 
end  
        
% Choose points at which to sample 15
x = (-1:.005:1); 
n = length(x); 

% Construct the covariance matrix 
C = zeros(n,n);
for i = 1:n 
    for j = 1:n 
        C(i,j)= k(x(i),x(j));
    end
end
R = 5; %generate Random Var-(u) R number of times
for c = 1:R;
% Sample from the Gaussian process at these 
u = randn(n,1); % sample u ~ N(0, I)
[A,S, B] = svd(C); % factor C = ASB'
z = A*sqrt(S)*u; % z = A S^.5 u 

% Plot 
figure(2); hold on; %clf   %dont forget to add/remove clf
plot(x,z)
axis([-1 1, -2, 2]) % remember to change the axis.
end