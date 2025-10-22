function output = glm_gaussian_v2
%% define some constants and initialze arrays
files  = dir('*.mat');
fs     = 1000;
d_fs   = 50;
ds_f   = fs / d_fs;
test   = 0.4;   
dilate = [[-1, 4]; [0, 2]; [0, 8]];
pred   = {'start_tone', 'correct_light', 'correct_poke', ...
    'incorrect_light', 'incorrect_poke', 'om_light', 'premature_poke', 'consumption'};

beh_train = []; % concat predictors for training
act_train = []; % concat response for training
beh_test  = []; % concat predictors for testing
act_test  = []; % concat response for testing
ons_test  = []; % concat behavior onsets for testing
dff       = []; % concat total response for shuffle
behav     = []; % concat total predictors for shuffle

%% load all the data in and prepare basis matrices
for i = 1:length(files)
    load(files(i).name)
    
    idx = src; idx.signal = [];
    signal = src.signal;
    
    onsets = glm_onsets(idx, length(signal));
    out    = nan(size(onsets, 1),  length(signal) / ds_f);
    for k = 1:size(onsets, 1); out(k, :) = binvec(onsets(k, :), ds_f); end

    win = gausswin(d_fs + 1, 12);  % full width of ~1 s
    win = win ./ sum(win);         % division here to prevent changes in amplitude
    signal = resample(signal, d_fs, fs);
    signal = conv(signal, win, 'same');
    behavior   = [];
    full_basis = [];
    full_lags  = [];
    
    for k = 1:size(out, 1)
        if contains(pred(k), ["light", "tone", "pellet"])
            [lags, basis] = glm_basis(out(k, :), dilate(2, :), 10, d_fs);
        elseif contains(pred(k), "poke")
            [lags, basis] = glm_basis(out(k, :), dilate(1, :), 10, d_fs);
        elseif contains(pred(k), "consumption")
            [lags, basis] = glm_basis(out(k, :), dilate(3, :), 10, d_fs);
        end
        full_basis = [full_basis; basis];
        full_lags  = [full_lags;  lags];
    end
    
    behavior = [behavior; full_basis];
    
    nframes  = size(signal, 2);
    train_fr = 1:(1 - test) * nframes;
    test_fr  = length(train_fr):nframes;

    beh_train = [beh_train behavior(:, train_fr)];
    beh_test  = [beh_test  behavior(:, test_fr)];
    act_train = [act_train signal(train_fr)];
    act_test  = [act_test  signal(test_fr)];

    ons_test  = [ons_test  out(:, test_fr)];

    dff   = [dff   signal];
    behav = [behav behavior];
end

%% let's run our GLM
fprintf('running GLM\n')

opts.alpha = 0.01;  % The regularization parameter
opts.standardize = true; 
options   = glmnetSet(opts);  % Set the options to use

% use sparse here to help with computation time
glmoutput = cvglmnet(sparse(beh_train'), act_train', 'gaussian', ...
                options, 'deviance', [], [], false, false, true);

% Default to the most regularized solution within 1 SE of the cross-validated minimum error
coeffs   = cvglmnetCoef(glmoutput, 'lambda_1se'); 
actpred  = cvglmnetPredict(glmoutput, beh_test', 'lambda_1se', 'response');
fraction = zeros(1, length(pred) + 1);
[fraction(1), ~, ~] = getDeviance(act_test, actpred, mean(act_train), 'Gaussian');

% light smoothing of predicted activity for visualization
act_pred = smooth(actpred);

%% get individual contributions for predictors within GLM and sum them
% idividual contributions for each predictor
% can leave out the intercept for this
idv_act = beh_test .* coeffs(2:end);

for i = 1:length(coeffs) - 1
    newcoeffs = coeffs(2:end);
    newcoeffs(setdiff(1:length(newcoeffs), i)) = 0;
    
    [fraction(i + 1), ~, ~] = getDeviance(act_test, idv_act(i, :), ...
                        mean(act_train), 'Gaussian');
end

%% recombine deviance
% remove negative fractions
fraction(fraction < 0) = 0;

% percent deviation for every predictor
explain = [fraction(1) 100 * fraction(2:end) / sum(fraction(2:end))];

% combine percentages across like predictors
sum_fraction = zeros(1, size(pred, 2) + 1);
sum_fraction(1) = fraction(1);

str = 2;
for i = 1:size(pred, 2)
    sum_fraction(i + 1) = sum(explain(str:str + full_lags(i) - 1));
%     fprintf('str idx: %f end idx: %f \n', str, str + full_lags(i) - 1)
    str = str + full_lags(i);
end

% devex = sum_fraction;
% devex(devex <= 0) = 0;
% for i = 2:length(sum_fraction); devex(i) = devex(i) ./ devex(1); end

% combine individual contributions for like predictors
sum_activity = zeros(size(pred, 2), size(idv_act, 2));
str = 1;
for i = 1:size(pred, 2)
    sum_activity(i, :) = smooth(sum(idv_act(str:str + full_lags(i) - 1, :)));
%     fprintf('str idx: %f end idx: %f \n', str, str + full_lags(i) - 1)
    str = str + full_lags(i);
end 

%% shuffle GLM by circular shifting dff
% shuffled glm
fprintf('shuffling GLM\n')

n_iter = 50;
for curr_boot = 1:n_iter
    time_range = 15 * d_fs;

    circshift_val = randsample(time_range:length(dff), 1);
    CVerr_boot    = cvglmnet(sparse(behav'), circshift(dff, circshift_val)', 'gaussian', ...
        options, 'deviance', [], [], false, false, true);
    coeff_shuff   = cvglmnetCoef(CVerr_boot);
    
    pred_test_shuff = cvglmnetPredict(CVerr_boot, beh_test', [], 'response');
    
    [A, B, C] = getDeviance(circshift(act_test, circshift_val), ...
        (pred_test_shuff), mean(dff), 'Gaussian');
    
    shuffled_glm{curr_boot}.GLM           = CVerr_boot;
    shuffled_glm{curr_boot}.circshift_val = circshift_val;
    shuffled_glm{curr_boot}.deviance      = [A B C];
end

%% put it all into a structure and save
output.fraction      = fraction;
output.coefficients  = coeffs;
output.dev_explain   = sum_fraction;
output.predictors    = pred;
output.pred_activity = act_pred;
output.test_behavior = beh_test;
output.test_activity = act_test;
output.idv_activity  = idv_act;
output.onsets        = ons_test;
output.shuffled      = shuffled_glm;
output.glm           = glmoutput;
output.sum_activity  = sum_activity;
output.lags          = full_lags;

% [path, name, ~] = fileparts(pwd);
% save([path '\glm_output\' name '_glm_output.mat'], 'output')
save bt268_5s_glm output
cd('..')

end

%% nested functions
function pred = glm_onsets(times, len)
    % collapse all tones into one array
    times.start_tone = sort([times.correct_tone; times.omitted_tone; times.premature_tone; times.incorrect_tone]);

    % initialize predictor array
    pred = zeros(8, len);

    % array
%     pred(1, floor(times.pellet_drop))     = 1;
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
    basis   = onsets;
    nframes = length(basis);

    % convolve events with a gaussian window
    win = gausswin(fs + 1, 12);  % full width of ~1 s
    win = win ./ sum(win);
    
    % Make it a basis matrix if necessary
    dilateframes = [round(dilate(1) * fs), round(dilate(2) * fs)];
    if dilateframes(1) == 0 && dilateframes(2) == 0
        basis = conv(basis, norm, 'same');
    else
        i = -spacing;
        while i >= dilateframes(1)
            newb = zeros(1, nframes);
            newb(1:end + i) = basis(1, -i + 1:end);
            basis = [basis; newb];
            i = i - spacing;
        end
        
        i = spacing;
        while i <= dilateframes(2)
            newb = zeros(1, nframes);
            newb(i + 1:end) = basis(1, 1:end - i);
            basis = [basis; newb];
            i = i + spacing;
        end
        
        for i = 1:size(basis, 1)
            basis(i, :) = conv(basis(i, :), win, 'same');
        end
    end
    
    lags = size(basis, 1);
end

function [frac, D_model, D_null] = getDeviance(y, yHat, mean_y_train, family)
    % [frac, D_model, D_null] = getDeviance(y, yHat, mean_y_train, family)
    % calculates the deviance for a variety of models.
    %
    % Inputs:
    % y                 Data 
    % yHat              Model prediction
    % mean_y_train      Mean of the training set of y (ensures that the null model does not get access to the test set).
    % family            Model distribution. "Poisson" and "Gaussian" are currently implemented.
    %
    % Outputs:
    % frac              The fraction of the null-model deviance that is explained by the model.
    % D_model           The residual deviance of the model.
    % D_null            The residual deviance of the null model, i.e. a model with only one free parameter (the mean of the data).
    
    if nargin < 4
        family = 'Poisson';
    end
    
    if nargin < 3 || isempty(mean_y_train)
        mean_y_train = mean(y);
    end
    
    y  = y(:);
    mu = yHat(:);
    
    switch family
        case 'Poisson'
            % Some useful sources:
            % https://en.wikipedia.org/wiki/Deviance_(statistics)
            % http://thestatsgeek.com/2014/04/26/deviance-goodness-of-fit-test-for-poisson-regression/
            % http://stats.stackexchange.com/questions/15730/poisson-deviance-and-what-about-zero-observed-values
                
            D_model = 2 * sum(nanRep(y .* log(y ./ mu), 0) + mu - y);
            D_null  = 2 * sum(nanRep(y .* log(y ./ mean_y_train), 0) + mean_y_train - y);
            
        case 'Gaussian'
            % This is simply the R-squared, https://en.wikipedia.org/wiki/Coefficient_of_determination
            D_model = sum((y - mu) .^ 2);
            D_null  = sum((y - mean_y_train') .^ 2);
    end
    
    frac = 1 - D_model/D_null;
%     figure; plot(y); hold on; plot(yHat)
end

function A = nanRep(A, fill)
    % A = nanRep(A, fill) replaces all NaNs in A with FILL. FILL can be empty,
    % but A will be converted to a linear array if FILL is empty.
    
    
    if isempty(fill)
        A(isnan(A)) = [];
    else
        A(isnan(A)) = fill;
    end
end
