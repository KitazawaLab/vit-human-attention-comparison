clear all
close all

load subj2subj_vit_gbvs_CW2019
%% struct for save
mds_res_all = struct;
embed_dim = 32; %2
%% 
mds_res = mdscale(subj, embed_dim, 'criterion','metricstress');%, Start="random");
mds_res_all.mds_res = mds_res;
%, 'criterion','metricsstress', Start="random"); % data include nan 
%scatter(mds_res(:, 1), mds_res(:, 2))
%% 
num_subj = length(subj);

%%  initial parameter for optimization
init_pos=zeros(1, embed_dim);
multi_start = true; % true; % 
if multi_start
    ms = MultiStart;
    num_ms = 10;
end
rng default % For reproducibility
options.Algorithm = 'levenberg-marquardt';
%% 
% gbvs
num_gbvs = 7;
model_name = "gbvs";
model_dist = eval(model_name);
mds_model = zeros(num_gbvs, embed_dim);
for model_idx = 1:num_gbvs
    disp(model_idx)
    subj_gbvs_dist = model_dist(:, model_idx);
    if multi_start
        problem = createOptimProblem('lsqnonlin', ...
            'objective', @(x)compute_error_model_subj(x, mds_res, subj_gbvs_dist), ...
            'x0',init_pos, 'options', options);
        [pos_min,fval] = run(ms,problem, num_ms);
    else
        pos_min = lsqnonlin(@(x)compute_error_model_subj(x, mds_res, subj_gbvs_dist), ...
            init_pos,[],[],options);
    end
    mds_model(model_idx, :) = pos_min;
end
mds_res_all.(model_name) = mds_model;

% ViT official model
arch_names = ["dino_deit_small16", "supervised_deit_small16"];
for model_name = arch_names
    model_dist = eval(model_name);
    disp(size(model_dist))
    size_model_dist = size(model_dist);
    depth = size_model_dist(2);
    num_heads = size_model_dist(3);
    mds_model = zeros(depth, num_heads, embed_dim);
    for depth_idx = 1:depth
        for head_idx = 1:num_heads
            disp([model_name, depth_idx, head_idx])
            subj_vit_dist = model_dist(:, depth_idx, head_idx);
            if multi_start
                problem = createOptimProblem('lsqnonlin', ...
                    'objective', @(x)compute_error_model_subj(x, mds_res, subj_vit_dist), ...
                    'x0',init_pos, 'options', options);
                [pos_min,fval] = run(ms,problem, num_ms);
            else
                pos_min = lsqnonlin(@(x)compute_error_model_subj(x, mds_res, subj_vit_dist), ...
                    init_pos,[],[],options);
            end
            mds_model(depth_idx, head_idx, :) = pos_min;
        end
    end
    mds_res_all.(model_name) = mds_model;
end

% ViT model
training_methods = ["dino", "supervised"];
depth_list = [4, 8, 12];
num_models = 6;
num_heads = 7;
for tm = training_methods
    for depth = depth_list
        disp(depth)
        model_name = strcat(tm, "_", num2str(depth));
        model_dist = eval(model_name);
        mds_model = zeros(num_models, depth, num_heads, embed_dim);
        for model_idx = 1:num_models
            for depth_idx = 1:depth
                for head_idx = 1:num_heads
                    disp([tm, depth, model_idx, depth_idx, head_idx])
                    subj_vit_dist = model_dist(:, model_idx, depth_idx, head_idx);
                    if multi_start
                        problem = createOptimProblem('lsqnonlin', ...
                            'objective', @(x)compute_error_model_subj(x, mds_res, subj_vit_dist), ...
                            'x0',init_pos, 'options', options);
                        [pos_min,fval] = run(ms, problem, num_ms);
                    else
                        pos_min = lsqnonlin(@(x)compute_error_model_subj(x, mds_res, subj_vit_dist), ...
                            init_pos,[],[],options);
                    end
                    mds_model(model_idx, depth_idx, head_idx, :) =  pos_min;
                end
            end
        end
        mds_res_all.(model_name) = mds_model;
    end
end

save('mds_results_CW2019_dim'+string(embed_dim)+'.mat', 'mds_res_all')

function stress_model_subj=compute_error_model_subj(a, mds_res, model_subj_distance)
    distance = vecnorm(bsxfun(@minus, mds_res, a), 2, 2);
    stress_model_subj = model_subj_distance-distance;
end