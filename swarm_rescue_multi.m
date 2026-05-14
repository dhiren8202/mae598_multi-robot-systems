function swarm_rescue_multi()
% Swarm Rescue (multi-target, single-file)
% - Slower motion
% - Random targets every run
% - Multiple targets with per-target fusion
% - Simple bounding-box collision avoidance (separation)

clc; clear; close all;

%% -------- Parameters --------
N = 10;                    % robots
M = 3;                     % targets (humans)
mapSize = [50 50];         % meters
dt = 0.1; T = 180;         % step, horizon

% Motion (SLOW)
v_search   = 0.30;         % m/s
v_converge = 0.55;         % m/s

% Sensing & estimation
detectRadius = 10;         % detection range (m)
Sigma_z = (0.6^2)*eye(2);  % measurement noise
tau_cov = 0.20;            % stop when all targets have trace(P) < tau_cov
Sigma_p = (0.01^2)*eye(2); % process noise (small)

% Collision avoidance via separation (bounding-box idea)
bbox = 0.8;                % side length of robot's bounding box (m)
sep_radius = 1.2*bbox;     % start repelling within this radius
k_sep = 0.8;               % repulsion strength (tune)

rng('shuffle');            % different targets each run

%% -------- World init --------
robots  = [rand(1,N)*mapSize(1); rand(1,N)*mapSize(2)];
targets = [rand(1,M)*mapSize(1); rand(1,M)*mapSize(2)];  % random each run

% Per-target information-form accumulators for WLS
Wsum = repmat(zeros(2), 1, 1, M);
bsum = repmat(zeros(2,1), 1, 1, M);
xhat = nan(2,M); P = repmat(nan(2), 1, 1, M);
known = false(1,M);              % becomes true after first detection

% Correlated random-walk memory
u_rw = 2*rand(2,N)-1;
last_refresh = zeros(1,N);

figure('Name','Swarm Rescue — Multi Target (slow + no-collide)');
hold on; axis equal; xlim([0 mapSize(1)]); ylim([0 mapSize(2)]);
grid on; box on;

%% -------- Main loop --------
for k = 1:round(T/dt)
    t = (k-1)*dt;

    % --- Decide goals: each robot moves toward nearest KNOWN target, else searches
    goals = nan(2,N);
    if any(known)
        % assign to nearest known target (by current estimate)
        X = xhat;         % 2 x M
        for i=1:N
            p = robots(:,i);
            d = vecnorm(X - p, 2, 1);
            d(~known) = inf;
            [~, j] = min(d);
            if isfinite(d(j))
                goals(:,i) = X(:,j);
            end
        end
    end

    % --- Desired velocity (search or converge)
    V_des = zeros(2,N);
    for i = 1:N
        if any(~isnan(goals(:,i)))
            dir = goals(:,i) - robots(:,i);
            V_des(:,i) = v_converge * safe_unit(dir);
        else
            % correlated random walk, refresh heading every 4 s
            if t - last_refresh(i) > 4
                u_rw(:,i) = 2*rand(2,1)-1; u_rw(:,i) = safe_unit(u_rw(:,i));
                last_refresh(i) = t;
            end
            V_des(:,i) = v_search * u_rw(:,i);
        end
    end

    % --- Separation (collision avoidance via bounding box)
    V_sep = zeros(2,N);
    for i = 1:N
        for j = i+1:N
            dvec = robots(:,i) - robots(:,j);
            d = norm(dvec);
            if d < sep_radius && d > 1e-6
                push = k_sep * (1/d - 1/sep_radius) * (dvec/d);
                V_sep(:,i) = V_sep(:,i) + push;
                V_sep(:,j) = V_sep(:,j) - push;
            end
        end
    end

    % --- Combine and integrate (cap speed by context)
    for i = 1:N
        v = V_des(:,i) + V_sep(:,i);
        vmax = any(~isnan(goals(:,i))) * v_converge + all(isnan(goals(:,i))) * v_search;
        n = norm(v); if n > vmax, v = v * (vmax/n); end
        robots(:,i) = robots(:,i) + dt*v + chol(Sigma_p)*randn(2,1);

        % keep within map
        robots(1,i) = min(max(robots(1,i), 0), mapSize(1));
        robots(2,i) = min(max(robots(2,i), 0), mapSize(2));
    end

    % --- Detections (per target)
    for i = 1:N
        for m = 1:M
            if norm(robots(:,i) - targets(:,m)) < detectRadius
                z = targets(:,m) + chol(Sigma_z)*randn(2,1);
                Wi = inv(Sigma_z); yi = Wi*z;
                Wsum(:,:,m) = Wsum(:,:,m) + Wi;
                bsum(:,:,m) = bsum(:,:,m) + yi;
                [xhat(:,m), P(:,:,m)] = wls_from_accum(Wsum(:,:,m), bsum(:,:,m));
                known(m) = true;
            end
        end
    end

    % --- Visualization
    cla; hold on;
    % draw targets (truth)
    plot(targets(1,:), targets(2,:), 'rx', 'MarkerSize', 12, 'LineWidth', 2);

    % draw robots + bounding boxes
    for i=1:N
        plot(robots(1,i), robots(2,i), 'bo', 'MarkerFaceColor','b', 'MarkerSize', 6);
        draw_bbox(robots(:,i), bbox);
    end

    % draw estimates & ellipses
    for m = 1:M
        if known(m)
            plot(xhat(1,m), xhat(2,m), 'k+', 'MarkerSize', 10, 'LineWidth', 2);
            draw_ellipse_95(xhat(:,m), P(:,:,m));
        end
    end

    title(sprintf('t = %.1fs | known targets: %d/%d', t, sum(known), M));
    drawnow limitrate;

    % --- Stop if all targets localized well
    if all(known)
        traces = arrayfun(@(m) trace(P(:,:,m)), 1:M);
        if all(traces < tau_cov)
            fprintf('SUCCESS at %.1fs. All targets localized (trace(P) < %.3f).\n', t, tau_cov);
            break;
        end
    end
end
end

%% -------- Helper functions --------
function u = safe_unit(v)
n = norm(v); if n < 1e-9, u = [0;0]; else, u = v/n; end
end

function [xhat, P] = wls_from_accum(Wsum, bsum)
% From accumulated information: Wsum = Σ Wi, bsum = Σ Wi*zi
xhat = Wsum \ bsum;
P = inv(Wsum);
end

function draw_ellipse_95(mu, P)
% Correct 95% confidence ellipse for 2D
S = (P + P')/2;
[V, D] = eig(S);
lam = max(diag(D), 1e-12);
R = V * diag(sqrt(lam));
th = linspace(0, 2*pi, 200);
circle = [cos(th); sin(th)];
E = R * circle * sqrt(5.991) + mu; % 5.991 = chi2inv(0.95,2)
plot(E(1,:), E(2,:), 'k-', 'LineWidth', 1.5);
end

function draw_bbox(p, side)
% Draw a square bounding box centered at position p
h = side/2;
x = [p(1)-h, p(1)+h, p(1)+h, p(1)-h, p(1)-h];
y = [p(2)-h, p(2)-h, p(2)+h, p(2)+h, p(2)-h];
plot(x, y, 'b-');
end
