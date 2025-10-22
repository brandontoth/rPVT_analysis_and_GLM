function cat_struct = make_new_pvt_struct(name, session, handle)

selpath = uigetdir();
filedir = selpath;
cd(filedir);

files  = dir('*.mat');

switch handle
    case 'dff'
        cat_struct = struct('triggerPellet', [], 'triggerPelletDrop', [], 'triggerCorrectLight', [], ...
            'triggerCorrectTone', [], 'triggerCorrectPoke', [], 'triggerIncorrectLight', [], ...
            'triggerIncorrectTone', [], 'triggerIncorrectPoke', [], 'triggerOmittedTone', [], ...
            'triggerOmittedLight', [], 'triggerPrematureTone', [], 'triggerPrematurePoke', []);
        fs = 1000;
    otherwise
        cat_struct = struct('triggerPellet', [], 'triggerCorrectLight', [], ...
            'triggerCorrectPoke', [], 'triggerIncorrectLight', [], 'triggerIncorrectPoke', [], ...
            'triggerOmittedLight', [], 'triggerPrematurePoke', [], 'triggerTone', []);
        fs = 50;
end

for i = 1:length(files)
    load(files(i).name)

    if exist('dat', 'var'); src = dat; end

%     pellet_drop = src.pellet_drop;
    pellet_ret  = src.pellet_ret;
    cor_light   = src.correct_light;
    cor_poke    = src.correct_poke;
    inc_light   = src.incorrect_light;
    inc_poke    = src.incorrect_poke;
    om_light    = src.omitted_light;
    pre_poke    = src.premature_poke;

    switch handle
        case 'dff'
            signal   = src.signal;
            cor_tone = src.correct_tone;
            inc_tone = src.incorrect_tone;
            om_tone  = src.omitted_tone;
            pre_tone = src.premature_tone;
            pellet_drop = src.pellet_drop;

        case 'test'
            signal = src.test_activity;
            tone   = src.str_tone;

        case 'pred'
            signal = src.pred_activity;
            tone   = src.str_tone;

        case 'correct poke'
            signal = src.pred_cor_poke;
            tone   = src.str_tone;

        case 'tone'
            signal = src.pred_tone;
            tone   = src.str_tone;

        case 'pellet'
            signal = src.pred_pell;
            tone   = src.str_tone;

        case 'consumption'
            signal = src.pred_cons;
            tone   = src.str_tone;

        case 'correct light'
            signal = src.pred_cor_light;
            tone   = src.str_tone;

        case 'incorrect light'
            signal = src.pred_inc_light;
            tone   = src.str_tone;

        case 'incorrect poke'
            signal = src.pred_inc_poke;
            tone   = src.str_tone;

        case 'omitted light'
            signal = src.pred_om_light;
            tone   = src.str_tone;

        case 'premature poke'
            signal = src.pred_pre_poke;
            tone   = src.str_tone;
    end

    switch handle
        case 'dff'
            for k = 1:length(pellet_drop)
                keep.triggerPelletDrop(k, :) = cutAroundEvent(pellet_drop(k), 10 * fs, signal);
            end

            for k = 1:length(cor_tone)
                keep.triggerCorrectTone(k, :) = cutAroundEvent(cor_tone(k), 10 * fs, signal);
            end

            for k = 1:length(inc_tone)
                keep.triggerIncorrectTone(k, :) = cutAroundEvent(inc_tone(k), 10 * fs, signal);
            end

            for k = 1:length(om_tone)
                keep.triggerOmittedTone(k, :) = cutAroundEvent(om_tone(k), 10 * fs, signal);
            end

            if ~isempty(pre_tone)
                for k = 1:length(pre_tone)
                    keep.triggerPrematureTone(k, :) = cutAroundEvent(pre_tone(k), 10 * fs, signal);
                end
            else
                keep.triggerPrematureTone = [];
            end

        otherwise
            for k = 1:length(tone)
                keep.triggerTone(k, :) = cutAroundEvent(tone(k), 10 * fs, signal);
            end
    end
    
    for k = 1:length(pellet_ret)
        keep.triggerPellet(k, :) = cutAroundEvent(pellet_ret(k), 10 * fs, signal);
    end

    for k = 1:length(cor_light)
        keep.triggerCorrectLight(k, :) = cutAroundEvent(cor_light(k), 10 * fs, signal);
    end

    for k = 1:length(cor_poke)
        keep.triggerCorrectPoke(k, :) = cutAroundEvent(cor_poke(k), 10 * fs, signal);
    end

    for k = 1:length(inc_light)
        keep.triggerIncorrectLight(k, :) = cutAroundEvent(inc_light(k), 10 * fs, signal);
    end

    for k = 1:length(inc_poke)
        keep.triggerIncorrectPoke(k, :) = cutAroundEvent(inc_poke(k), 10 * fs, signal);
    end

    for k = 1:length(om_light)
        keep.triggerOmittedLight(k, :) = cutAroundEvent(om_light(k), 10 * fs, signal);
    end

    if ~isempty(pre_poke)
        for k = 1:length(pre_poke)
            keep.triggerPrematurePoke(k, :) = cutAroundEvent(pre_poke(k), 10 * fs, signal);
        end
    else
        keep.triggerPrematurePoke = [];
    end

    my_struct_fields = fieldnames(cat_struct);
    super_struct = arrayfun(@(i) [cat_struct.(my_struct_fields{i}); ...
        keep.(my_struct_fields {i})], (1:numel(my_struct_fields))', 'un', 0);
    my_dirty_trick = [my_struct_fields, super_struct]';
    cat_struct   = struct(my_dirty_trick{:});

    clear src keep dat
end

% cd('..')
save([name '_' session '.mat'], 'cat_struct')

end