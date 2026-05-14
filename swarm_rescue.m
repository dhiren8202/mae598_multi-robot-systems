function swarm_rescue()
% Simple Swarm Rescue (single file)
% Robots wander, detect a human, fuse tentative coordinates (WLS),
% and stop when uncertainty is small enough.

clc; clear; close all;

% ---------- Parameters ----------
N = 8;                        % number of robots
mapSize = [50 50];            % meters
dt = 0.1;                     % time step
T  = 180;                     % total sim seconds
detectRadius = 12;            % detection range (m)
Sigma_z = (0.8^2)*eye(2);     % measurement noise covariance
Sigma_p = (0.02^2)*eye(2);    % motion noise covariance
tau_cov = 0.25;               % stop if trace(P) < tau_cov

% ---------- World init ----------
rng(1);                       % reproducible motion
robots = [rand(1,N)*mapSize(1); rand(1,N)*mapSize(2)];
human  = [35; 18];            % ground truth
centroid_guess = [NaN; NaN];  % becomes available after first detection
mode = repmat("Search",1,N);  % "Search" or "Respond"

% ---------- Figure ----------
figure('Name','Swarm Rescue');
hold on; axis equal;
xlim([0 mapSize(1)]); ylim([0 mapSize(2)]);
grid on; box on;

% Predeclare to avoid “undefined variable” warnings
xhat = []; P = [];

% ---------- Main loop ----------
for t = 0:dt:T
    detections = [];  % 2xM matrix of tentative coordinates this step

    % 1) Move robots and check detections
    for i = 1:N
        p = robots(:,i);

        if mode(i) == "Search"
            v = 0.8 * (2*rand(2,1) - 1);  % random walk
        else
            if any(isnan(centroid_guess))
                v = [0;0];
            else
                dir = centroid_guess - p;
                v = 1.0 * dir / (norm(dir) + 1e-6);
            end
        end

        % integrate motion + small noise
        robots(:,i) = p + dt*v + chol(Sigma_p)*randn(2,1);

        % keep inside bounds
        robots(1,i) = min(max(robots(1,i), 0), mapSize(1));
        robots(2,i) = min(max(robots(2,i), 0), mapSize(2));

        % detection -> make a noisy tentative coordinate
        if norm(robots(:,i) - human) < detectRadius
            z = human + chol(Sigma_z)*randn(2,1);
            detections = [detections, z]; %#ok<AGROW>
            mode(:) = "Respond";          % everyone starts converging
            centroid_guess = z;           % coarse attractor for motion
        end
    end

    % 2) Fuse detections (weighted least squares) if any
    if ~isempty(detections)
        [xhat, P] = wls_fuse(detections, Sigma_z);
        centroid_guess = xhat; % move toward the fused estimate
    end

    % 3) Draw
    cla; hold on;
    plot(robots(1,:), robots(2,:), 'bo', 'MarkerFaceColor','b');
    plot(human(1), human(2), 'rx', 'MarkerSize',12, 'LineWidth',2);
    if ~isempty(xhat)
        plot(xhat(1), xhat(2), 'k+', 'MarkerSize',10, 'LineWidth',2);
        draw_ellipse_95(xhat, P);
    end
    title(sprintf('t = %.1f s, detections = %d', t, size(detections,2)));
    drawnow;

    % 4) Stop condition
    if ~isempty(P) && trace(P) < tau_cov
        fprintf('Localized at [%.2f, %.2f] in %.1f s (trace(P)=%.3f)\n', ...
            xhat(1), xhat(2), t, trace(P));
        break;
    end
end
end

% ---------- Helpers (local functions) ----------
function [xhat, P] = wls_fuse(Z, Sigma)
% Z is 2xM, Sigma is 2x2 covariance for each measurement (same here)
W = zeros(2,2);
b = zeros(2,1);
Wi = inv(Sigma);
for i = 1:size(Z,2)
    W = W + Wi;
    b = b + Wi * Z(:,i);
end
xhat = W \ b;       % solve W*x = b
P = inv(W);         % fused covariance
end

function draw_ellipse_95(mu, P)
% Draw 95% confidence ellipse for 2D Gaussian
S = (P + P')/2;               % symmetrize for numeric stability
[V, D] = eig(S);
lam = max(diag(D), 1e-12);    % clamp to avoid negatives
R = V * diag(sqrt(lam));      % square-root of covariance
theta = linspace(0, 2*pi, 200);
circle = [cos(theta); sin(theta)];
E = R * circle * sqrt(5.991) + mu;   % 5.991 = chi2inv(0.95,2)
plot(E(1,:), E(2,:), 'k-', 'LineWidth', 1.5);
end
