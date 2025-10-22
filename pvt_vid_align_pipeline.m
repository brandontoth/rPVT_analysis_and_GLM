%% Full script to align LabGym video data to FP
% date: 251015
% author: BT

%% Step 1: Grab session FP/FED data
% clear, clc, close all

% define file locations
user     = getenv('USERPROFILE');
root_dir = [user  '\University of Michigan Dropbox' ...
    '\MED-burgesslab\BurgessLab_data_transfer\Brandon\2 Vigilance\'];

[nidaq_name, nidaq_folder] = uigetfile('*.*', 'Choose NIDAQ file', root_dir);
NIDAQ = fullfile(nidaq_folder, nidaq_name);         

[file_name, file_folder]   = uigetfile('*.*', 'Choose CSV file', root_dir);
filePath = fullfile(file_folder, file_name);

% folder to save to
data_folder = [user 'University of Michigan Dropbox\MED-burgesslab\' ...
    'BurgessLab_data_transfer\Brandon\2 Vigilance\4 spring 2025\2 FP'];

% constant declaration
base_adj = 0.1;
cut_time = 10000;
sync_ch  = 2;
percentile = 10;
ttl_ch     = 3;
fs = 1000;

% load NIDAQ data
load(NIDAQ);

% load FP
[fp_name, fp_folder] = uigetfile({[root_dir '\*.*']}, 'Select PPD'); % select ppd of interest
fp_path = fullfile(fp_folder, fp_name);                                   % generate file path

fp_file = import_ppd(fp_path); % load data

% assign data to respective variables
fp_right = fp_file.analog_1;
fp_left  = fp_file.analog_2;
sync     = fp_file.digital_1;
fp_fs       = fp_file.sampling_rate;

% sanity check
figure; plot(smooth(fp_right, 20), 'r'); hold on; plot(smooth(fp_left, 20), 'k');

% Resample FP data to match NIDAQ
fp_r_long = resample(fp_right, fs, fp_fs);
fp_l_long = resample(fp_left,  fs, fp_fs);
sync_long = resample(+sync,    fs, fp_fs);

% Sync FP to NIDAQ and plot 
[~, ~, z_fp_sig] = align_fp_w_isos(sync_long, data(:, sync_ch), ...
    fp_r_long, fp_l_long, base_adj);

% align CSV to NIDAQ
fed_data  = alignNIDAQ(filePath, data, ttl_ch);

% cut around relevant task events
[~, triggers_isos] = getFedTriggers(fed_data, timeStamps, z_fp_sig, cut_time, fs);

% plot everything
plotAllVigilanceOutcomesWithFP(triggers_isos, cut_time, ...
    '', '', 0, 'baseline', fs, [-1 2])

%% Step 2: Reconstruct TTL from video
close all; clc

% === LOAD VIDEO ===
[fn, pn] = uigetfile({'*.mp4;*.avi;*.mov','Video Files'}, 'Select video');
if isequal(fn, 0), return; end
vid    = VideoReader(fullfile(pn, fn));
cam_fs = vid.NumFrames / (length(z_fp_sig) / fs);

fprintf('Loaded %s (nominal %.2f fps)\n', fn, vid.FrameRate);

% === SELECT TWO ROIs ===
frame1 = read(vid, 500); % just need a frame where you can see the FED clearly
imshow(frame1);
title('Draw three ROIs (drag box, double-click inside to confirm)');

disp('Draw ROI 1...');
roi1 = drawrectangle('Label', 'ROI 1'); wait(roi1);
disp('Draw ROI 2...');
roi2 = drawrectangle('Label', 'ROI 2'); wait(roi2);
disp('Draw ROI 3...');
roi3 = drawrectangle('Label', 'background'); wait(roi3);

pos1 = round(roi1.Position); % [x y width height]
pos2 = round(roi2.Position); % [x y width height]
pos3 = round(roi3.Position); % [x y width height]
close(gcf);

fprintf('ROI 1: [x=%d, y=%d, w=%d, h=%d]\n', pos1);
fprintf('ROI 2: [x=%d, y=%d, w=%d, h=%d]\n', pos2);
fprintf('ROI 3: [x=%d, y=%d, w=%d, h=%d]\n', pos2);

% === EXTRACT INTENSITY VALUES ===
vid = VideoReader(fullfile(pn,fn)); % rewind
light1 = [];
light2 = [];
backgrnd = [];
times    = [];

while hasFrame(vid)
    f = readFrame(vid);
    t = vid.CurrentTime;
    times(end + 1, 1) = t;

    if size(f, 3) == 3
        f = rgb2gray(f);
    end

    roi1_pix = f(pos1(2):pos1(2) + pos1(4) - 1, pos1(1):pos1(1) + pos1(3) - 1);
    roi2_pix = f(pos2(2):pos2(2) + pos2(4) - 1, pos2(1):pos2(1) + pos2(3) - 1);
    roi3_pix = f(pos3(2):pos3(2) + pos3(4) - 1, pos3(1):pos3(1) + pos3(3) - 1);

    light1  (end + 1, 1) = mean(roi1_pix(:));
    light2  (end + 1, 1) = mean(roi2_pix(:));
    backgrnd(end + 1, 1) = mean(roi3_pix(:));
end
 
times = times - times(1);

% Combine lights and subtract background
composite = (light1 + light2) / 2;
composite = composite - backgrnd; figure; plot(composite)

% remove non-linear trends
opol = 16;
t = 1:length(composite);
[p, s, mu] = polyfit(t, composite', opol);
f_y = polyval(p, t, [], mu);

dt_comp = composite' - f_y;

figure; tiledlayout(2, 1);
a = nexttile; plot(composite); hold on; plot(f_y, 'LineWidth', 2)
b = nexttile; plot(dt_comp);
linkaxes([a b], 'x')

% Binarize composite into a TTL trace
vid_ttl = zeros(length(dt_comp), 1);
vid_ttl(dt_comp > 20) = 5; figure; plot(vid_ttl)

% Interpolate video TTLs to match NIDAQ
t_original = (0:numel(vid_ttl) - 1) / cam_fs;
% Target time vector (1000 Hz)
t_target = 0:1/1000:t_original(end);
% Interpolate using shape-preserving cubic interpolation
interp_ttl = interp1(t_original, vid_ttl, t_target, 'pchip');

% Have to binarize again due to sampling artifact from interpolation
ttl_bin = zeros(length(interp_ttl), 1);
ttl_bin(interp_ttl > 0) = 5; figure; plot(ttl_bin)

%% Step 3: Clean up TTL trace
merge_gap_sec = 2;      % merge if gaps < 2 s
merge_gap_samples = merge_gap_sec * fs;

% === FIND HIGH SEGMENTS ===
is_high   = ttl_bin > 0;            % logical vector
diff_high = diff([0; is_high; 0]);  % pad to detect edges
onsets  = find(diff_high == 1);
offsets = find(diff_high == -1) - 1;

% === MERGE NEARBY SEGMENTS ===
merged = false(size(is_high));
if isempty(onsets)
    warning('No TTL pulses detected.');
else
    merged_onsets = onsets(1);
    merged_offsets = offsets(1);

    for i = 2:numel(onsets)
        gap = onsets(i) - merged_offsets(end);

        if gap <= merge_gap_samples
            % Merge: extend last segment
            merged_offsets(end) = offsets(i);
        else
            % New independent segment
            merged_onsets (end + 1)  = onsets(i);
            merged_offsets(end + 1) = offsets(i);
        end
    end

    % Fill merged TTL high periods
    for i = 1:numel(merged_onsets)
        merged(merged_onsets(i):merged_offsets(i)) = true;
    end
end

% === RECREATE CLEAN TTL TRACE ===
ttl_clean = zeros(size(ttl_bin));
ttl_clean(merged) = 5;

% sanity check
t = (0:numel(ttl_bin) - 1) / fs;
figure;
plot(t, ttl_bin,   'k', 'DisplayName', 'Original');
hold on;
plot(t, ttl_clean, 'r', 'LineWidth', 1.5, 'DisplayName', 'Cleaned TTL');
xlabel('Time (s)');
ylabel('TTL (V)');
legend;
title(sprintf('TTL cleaned with %.1f s merge window', merge_gap_sec));

%% Step 4: Create the TTLs of lights from the NIDAQQ
n_samps  = length(z_fp_sig);
daq_ttls = false(n_samps, 1);

left_idx  = floor(fed_data.datetimeSync(fed_data.event == "Left light"));
right_idx = floor(fed_data.datetimeSync(fed_data.event == "Right light"));

% Keep only indices within valid range
left_idx (left_idx  < 1 | left_idx  > n_samps) = [];
right_idx(right_idx < 1 | right_idx > n_samps) = [];

% Assign logicals
daq_ttls(left_idx)  = true;
daq_ttls(right_idx) = true;

%% Step 5: Get the velocity data from LabGym
% string that we're looking for in filenames
target    = 'Mouse_velocity.xlsx';

% get desired folders in the current directory
D = dir;
D = D(~ismember({D.name}, {'.', '..'}));
D = D([D(:).isdir]);

% loop through folders and get velocity data
for k = 1:numel(D)
    currD = D(k).name; % get the current subdirectory name
    cd(currD)          % change the directory

    fprintf(1, 'Now reading %s\n', currD);

    velo_file = dir;
    velo_file = velo_file(ismember({velo_file.name}, {target}));
    velo = readtable(velo_file.name);

    all_velo(:, k) = velo{:, 2};

    clear velo_file velo

    cd('..')
end

% remove nans and combine across columns
all_velo(isnan(all_velo)) = 0;
all_velo = sum(all_velo, 2);

% convolve with gaussian window to smooth data
sigma_s = 0.005;        % 2 ms
sigma = fs * sigma_s;  % 10 samples

N = round(6 * sigma);  % window length ~ +/-3σ
alpha = (N - 1) / (2 * sigma);

win = gausswin(N, alpha);
win = win ./ sum(win);

conv_velo = conv(all_velo, win, 'same');

% sanity check
figure; plot(conv_velo); xlim([0 length(conv_velo)])

% Interpolate to match NIDAQ sampling rate
% light_interp = interp1(times, my_light, t_target, 'pchip', 'extrap');
t_original = (0:numel(conv_velo) - 1) / cam_fs;
% Target time vector (1000 Hz)
t_target = 0:1/1000:t_original(end);
% Interpolate using shape-preserving cubic interpolation
interp_velo = interp1(t_original, conv_velo, t_target, 'pchip');

%% Step 6: Align everything
vid_diff = find(diff(ttl_clean) > 0.1);
daq_diff = find(diff(daq_ttls)  > 0.1);

offset = daq_diff(1) - vid_diff(1);
align_velo = [zeros(offset, 1); interp_velo'];
align_velo = align_velo(1:length(z_fp_sig));

align_ttl = [zeros(offset, 1); ttl_clean];
align_ttl = align_ttl(1:length(z_fp_sig));

figure; hold on; plot(daq_ttls * 5); plot(align_ttl)

%% Step 7 (optional): Shrink/stretch alignment if needed
stretch = 0.999;  % adjust this factor (e.g., 1.001 ≈ +1s drift over ~1000s)
N = numel(align_ttl);
t = (0:N - 1)' / fs;

% Warp time axis
t_warp = t * stretch;

% Resample back to original time base (same number of samples)
ttl_stretched = interp1(t_warp, double(align_ttl), t, 'linear', 'extrap') > 0.5;
stretch_velo  = interp1(t_warp, align_velo, t, 'linear', 'extrap');

figure; hold on
plot(daq_ttls)
plot(ttl_stretched)

%% Step 8: Save FP and aligned velocity
save photometry z_fp_sig
save velocity stretch_velo

%% Plotting full trace
test = nan(1, length(z_fp_sig));
for i = 1:length(fed_data.datetimeSync) - 1 % edge case for when two events have the same time
    if fed_data.datetimeSync(i) == fed_data.datetimeSync(i + 1)
        fed_data.datetimeSync(i + 1) = fed_data.datetimeSync(i + 1) + 1;
    end
end

j = 1;
for i = 1:length(test)
    if i == round(fed_data.datetimeSync(j))
        if fed_data.event(j) == "Tone"
            test(i) = 0;
        elseif contains(fed_data.event(j), "light")
            test(i) = 1;
        elseif fed_data.event(j) == "Right" || fed_data.event(j) == "Left"
            test(i) = 2;
        elseif fed_data.event(j) == "Pellet"
            test(i) = 3;
        end
        j = j + 1;
    end
end

figure;
plot(smooth(z_fp_sig, 2000), 'Color', '#090088','LineWidth',1.5)
hold on;
plot(test, 'o')
plot(zscore(stretch_velo),'Color','#808080')
ylim([-3 5])

%% Plot velocity around task events
% cut around relevant task events
[~, triggers_isos] = getFedTriggers(fed_data, timeStamps, ...
    zscore(stretch_velo), cut_time, fs);

% plot everything
plotAllVigilanceOutcomesWithFP(triggers_isos, cut_time, ...
    '', '', 0, 'Velocity', fs, [-1 2])

%% Plot velocity/FP cross-correlation
[corr, lag] = xcorr(zscore(stretch_velo), z_fp_sig, 100 * fs, "normalized");
figure; plot(lag / Fs, corr);
box off
pbaspect([1 1 1])
ax = gca; ax.FontSize = 12; 
xlabel("Time (s)", "FontSize", 12)
ylabel("Norm. correlation", "FontSize", 12)