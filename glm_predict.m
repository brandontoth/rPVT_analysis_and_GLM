function dat = glm_predict

% get relevant glm output file
[file_name, file_folder] = uigetfile('', 'Choose GLM output file', '*.*');
GLM = fullfile(file_folder, file_name);
load(GLM) %#ok<LOAD> 

% pellet_drop = output.onsets(1, :); dat.pellet_drop     = find(diff(pellet_drop) == 0.05);
pellet_ret  = output.onsets(8, :); dat.pellet_ret      = find(diff(pellet_ret)  == 0.05);
str_tone    = output.onsets(1, :); dat.str_tone        = find(diff(str_tone)    == 0.05);
cor_light   = output.onsets(2, :); dat.correct_light   = find(diff(cor_light)   == 0.05);
cor_poke    = output.onsets(3, :); dat.correct_poke    = find(diff(cor_poke)    == 0.05);
inc_light   = output.onsets(4, :); dat.incorrect_light = find(diff(inc_light)   == 0.05);
inc_poke    = output.onsets(5, :); dat.incorrect_poke  = find(diff(inc_poke)    == 0.05);
om_light    = output.onsets(6, :); dat.omitted_light   = find(diff(om_light)    == 0.05);
pre_poke    = output.onsets(7, :); dat.premature_poke  = find(diff(pre_poke)    == 0.05);

fields = string(fieldnames(dat));
for i = 1:length(fields)
    dat.(fields{i})(dat.(fields{i}) < 500) = [];
end

dat.test_activity  = output.test_activity;
dat.pred_activity  = output.pred_activity';
dat.pred_cor_poke  = output.sum_activity(3, :);
% dat.pred_pell      = output.sum_activity(1, :);
dat.pred_cons      = output.sum_activity(8, :);
dat.pred_tone      = output.sum_activity(1, :);
dat.pred_cor_light = output.sum_activity(2, :);
dat.pred_inc_light = output.sum_activity(4, :);
dat.pred_inc_poke  = output.sum_activity(5, :);
dat.pred_om_light  = output.sum_activity(6, :);
dat.pred_pre_poke  = output.sum_activity(7, :);

save([file_name(1:5) '_pred.mat'], 'dat')

end



% 
% 
% for k = 2:length(cor_poke)
%    dat.triggerCorrectPoke(k, :) = cutAroundEvent(cor_poke(k), 15 * fs, signal);
%    dat.pred_poke_cor_poke(k, :) = cutAroundEvent(cor_poke(k), 15 * fs, pred_poke);
%    dat.pred_pell_cor_poke(k, :) = cutAroundEvent(cor_poke(k), 15 * fs, pred_pell);
%    dat.pred_cons_cor_poke(k, :) = cutAroundEvent(cor_poke(k), 15 * fs, pred_cons); 
%    dat.pred_cor_poke(k, :) = cutAroundEvent(cor_poke(k), 15 * fs, pred_sig);
% end
% 
% % figure; hold on;
% % plot(mean(dat.triggerCorrectPoke));
% % plot(mean(dat.pred_poke_cor_poke)); 
% % plot(mean(dat.pred_pell_cor_poke)); 
% % plot(mean(dat.pred_cons_cor_poke)); 
% % plot(mean(dat.pred_cor_poke))
% 
% % save('BT179_pred_poke.mat', 'dat')
% 
% %%
% fs = 50;
% figure;
% shadedErrorBar([], dat.triggerCorrectPoke, {@mean, @(x) std(x) / sqrt(size(x, 1))})
% % shadedErrorBar([], dat.pred_poke_cor_poke, {@mean, @(x) std(x) / sqrt(size(x, 1))}, 'lineProps', 'b')
% % shadedErrorBar([], dat.pred_pell_cor_poke, {@mean, @(x) std(x) / sqrt(size(x, 1))}, 'lineProps', 'g')
% % shadedErrorBar([], dat.pred_cons_cor_poke, {@mean, @(x) std(x) / sqrt(size(x, 1))}, 'lineProps', 'r')
% shadedErrorBar([], dat.pred_cor_poke, {@mean, @(x) std(x) / sqrt(size(x, 1))}, 'lineProps', 'm')
% pbaspect([1 1 1])
% ylabel('z-\DeltaF/F')
% xlabel('Time from poke (s)')
% xlim([0 size(dat.pred_cor_poke, 2)])
% xline(fs * 15, 'Color',[1 1 1] * 0.7,'LineStyle',':', 'LineWidth', 1);
% xt = get(gca, 'xtick');
% set(gca, 'XTick', xt, 'xticklabel', (xt - fs * 15) / fs)
% legend('\color{black} Real', '\color{magenta} Predicted')
% legend('boxoff')
% 
% 
% % plot(mean(dat.triggerCorrectPoke));
% % plot(mean(dat.pred_poke_cor_poke)); 
% % plot(mean(dat.pred_pell_cor_poke)); 
% % plot(mean(dat.pred_cons_cor_poke)); 
% % plot(mean(dat.pred_cor_poke))