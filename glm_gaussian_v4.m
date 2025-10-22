function output = glm_gaussian_v4
%% define constants and initialize arrays
files  = dir('*.mat');
fs     = 1000;
d_fs   = 50;
ds_f   = fs / d_fs;
test   = 0.4;   
dilate = [[-1, 4]; [0, 2]; [0, 8]];
pred   = {'start_tone', 'correct_light', 'correct_poke', ...
          'incorrect_light', 'incorrect_poke', 'om_light', ...
          'premature_poke', 'consumption', 'velocity'};  % added velocity

beh_train = []; % predictors for training
act_train = []; % response for training
beh_test  = []; % predictors for testing
act_test  = []; % response for testing
ons_test  = []; % behavior onsets for testing
dff       = []; % total response for shuffle
behav     = []; % total predictors for shuffle
lag_counts_all = []; % lag counts for each predictor (excluding velocity)

%% load data and prepare basis matrices
for i = 1:length(files)
    load(files(i).name, 'src');

    idx = src; 
    signal = src.signal;

    onsets = glm_onsets(idx, length(signal));
    out    = nan(size(onsets, 1), ceil(length(signal) / ds_f));
    for k = 1:size(onsets, 1)
        out(k, :) = binvec(onsets(k, :), ds_f);
    end

    % light smoothing of dFF
    win = gausswin(d_fs + 1, 12);  % ~1 s window
    win = win ./ sum(win);
    signal = resample(signal, d_fs, fs);
    signal = conv(signal, win, 'same');

    behavior   = [];
    full_basis = [];
    lag_counts = [];  % per predictor lag count (not including velocity)

    % ----- existing predictors -----
    for k = 1:size(out, 1)
        if contains(pred{k}, ["light", "tone", "pellet"])
            [n_lags, basis] = glm_basis(out(k, :), dilate(2, :), 10, d_fs);
        elseif contains(pred{k}, "poke")
            [n_lags, basis] = glm_basis(out(k, :), dilate(1, :), 10, d_fs);
        elseif contains(pred{k}, "consumption")
            [n_lags, basis] = glm_basis(out(k, :), dilate(3, :), 10, d_fs);
        end
        full_basis = [full_basis; basis];
        lag_counts = [lag_counts; n_lags];
    end

    % ----- add velocity predictor if it exists in this session -----
    if isfield(src, 'velocity') && ~isempty(src.velocity)
        vel = resample(src.velocity, d_fs, fs);   % downsample
        vel = zscore(vel);                        % z-score
        full_basis = [full_basis; vel];           % append as one row
        lag_counts = [lag_counts; 1];             % velocity has 1 lag
    else
        full_basis = [full_basis; zeros(1, size(full_basis, 2))]; % fill zeros
        lag_counts = [lag_counts; 1];
    end

    behavior = [behavior; full_basis];

    nframes  = size(signal, 2);
    train_fr = 1:floor((1 - test) * nframes);
    test_fr  = train_fr(end) + 1:nframes;

    beh_train = [beh_train behavior(:, train_fr)];
    beh_test  = [beh_test  behavior(:, test_fr)];
    act_train = [act_train signal(train_fr)];
    act_test  = [act_test  signal(test_fr)];

    ons_test  = [ons_test  out(:, test_fr)];

    dff   = [dff   signal];
    behav = [behav behavior];
    lag_counts_all = [lag_counts_all; lag_counts];
end

%% run GLM
fprintf('Running GLM...\n')

opts.alpha = 0.01;  
opts.standardize = true; 
options = glmnetSet(opts);

glmoutput = cvglmnet(sparse(beh_train'), act_train', 'gaussian', ...
                     options, 'deviance', [], [], false, false, true);

coeffs   = cvglmnetCoef(glmoutput, 'lambda_1se'); 
actpred  = cvglmnetPredict(glmoutput, beh_test', 'lambda_1se', 'response');
fraction = zeros(1, length(pred) + 1);
[fraction(1), ~, ~] = getDeviance(act_test, actpred, mean(act_train), 'Gaussian');

act_pred = smooth(actpred);

%% compute individual predictor contributions
idv_act = beh_test .* coeffs(2:end);

for i = 1:length(coeffs) - 1
    newcoeffs = coeffs(2:end);
    newcoeffs(setdiff(1:length(newcoeffs), i)) = 0;
    [fraction(i + 1), ~, ~] = getDeviance(act_test, idv_act(i, :), ...
                                mean(act_train), 'Gaussian');
end

fraction(fraction < 0) = 0;
explain = [fraction(1) 100 * fraction(2:end) / sum(fraction(2:end))];

% combine lag contributions per predictor
sum_fraction = zeros(1, numel(pred) + 1);
sum_fraction(1) = fraction(1);

start_idx = 2;
for i = 1:numel(pred)
    end_idx = start_idx + lag_counts_all(i) - 1;
    sum_fraction(i + 1) = sum(explain(start_idx:end_idx));
    start_idx = end_idx + 1;
end

sum_activity = zeros(numel(pred), size(idv_act, 2));
start_idx = 1;
for i = 1:numel(pred)
    end_idx = start_idx + lag_counts_all(i) - 1;
    sum_activity(i, :) = smooth(sum(idv_act(start_idx:end_idx, :)));
    start_idx = end_idx + 1;
end

%% shuffle control
fprintf('Running shuffled GLM (50 iterations)...\n')
n_iter = 50;
shuffled_glm = cell(1, n_iter);

for curr_boot = 1:n_iter
    time_range = 15 * d_fs;
    circshift_val = randsample(time_range:length(dff), 1);

    CVerr_boot = cvglmnet(sparse(behav'), circshift(dff, circshift_val)', ...
                          'gaussian', options, 'deviance', [], [], false, false, true);
    coeff_shuff   = cvglmnetCoef(CVerr_boot);
    pred_test_shuff = cvglmnetPredict(CVerr_boot, beh_test', [], 'response');

    [A, B, C] = getDeviance(circshift(act_test, circshift_val), pred_test_shuff, ...
                            mean(dff), 'Gaussian');

    shuffled_glm{curr_boot} = struct( ...
        'GLM', CVerr_boot, ...
        'coeffs', coeff_shuff, ...
        'circshift_val', circshift_val, ...
        'deviance', [A B C]);
end

%% output
output = struct( ...
    'fraction', fraction, ...
    'coefficients', coeffs, ...
    'dev_explain', sum_fraction, ...
    'predictors', {pred}, ...
    'pred_activity', act_pred, ...
    'test_behavior', beh_test, ...
    'test_activity', act_test, ...
    'idv_activity', idv_act, ...
    'onsets', ons_test, ...
    'shuffled', {shuffled_glm}, ...
    'glm', glmoutput, ...
    'sum_activity', sum_activity, ...
    'lags', lag_counts_all);

[path, name, ~] = fileparts(pwd);
save([path '\glm_output\' name '_glm_output.mat'], 'output')

fprintf('GLM complete. Saved to %s\n', fullfile(out_dir, [name '_glm_output.mat']));
cd('..')

end


%% ----------- NESTED FUNCTIONS -----------
function pred = glm_onsets(times, len)
    times.start_tone = sort([times.correct_tone; times.omitted_tone; ...
                             times.premature_tone; times.incorrect_tone]);

    pred = zeros(8, len);
    pred(1, floor(times.start_tone))      = 1;
    pred(2, floor(times.correct_light))   = 1;
    pred(3, floor(times.correct_poke))    = 1;
    pred(4, floor(times.incorrect_light)) = 1;
    pred(5, floor(times.incorrect_poke))  = 1;
    pred(6, floor(times.omitted_light))   = 1;
    pred(7, floor(times.premature_poke))  = 1;
    pred(8, floor(times.pellet_ret))      = 1;
end


function [lags, basis] = glm_basis(onsets, dilate, spacing, fs)
    nframes = length(onsets);
    win = gausswin(fs + 1, 12);
    win = win ./ sum(win);

    dilateframes = round(dilate * fs);
    lags_vec = dilateframes(1):spacing:dilateframes(2);

    basis = zeros(length(lags_vec), nframes);
    for j = 1:length(lags_vec)
        shifted = circshift(onsets, lags_vec(j));
        basis(j, :) = conv(shifted, win, 'same');
    end
    lags = length(lags_vec);
end


function [frac, D_model, D_null] = getDeviance(y, yHat, mean_y_train, family)
    if nargin < 4, family = 'Poisson'; end
    if nargin < 3 || isempty(mean_y_train), mean_y_train = mean(y); end
    y = y(:); mu = yHat(:);

    switch family
        case 'Poisson'
            D_model = 2 * sum(nanRep(y .* log(y ./ mu), 0) + mu - y);
            D_null  = 2 * sum(nanRep(y .* log(y ./ mean_y_train), 0) + mean_y_train - y);
        case 'Gaussian'
            D_model = sum((y - mu).^2);
            D_null  = sum((y - mean_y_train).^2);
    end
    frac = 1 - D_model / D_null;
end


function A = nanRep(A, fill)
    if isempty(fill)
        A(isnan(A)) = [];
    else
        A(isnan(A)) = fill;
    end
end
