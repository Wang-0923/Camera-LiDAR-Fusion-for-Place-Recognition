function export_relja_netvlad_vlad_params(input_mat, output_mat)
% Export NetVLAD cluster parameters from Relja's MatConvNet .mat model.
%
% Usage:
%   matlab -batch "addpath('tools'); export_relja_netvlad_vlad_params('/path/model.mat','/path/vlad_params.mat')"

data = load(input_mat, 'net');
net = data.net;
clsts = [];
clstsAssign = [];

for i = 1:numel(net.layers)
    layer = net.layers{i};
    candidates = {};
    if isfield(layer, 'forward')
        candidates{end + 1} = layer.forward; %#ok<AGROW>
    end
    if isfield(layer, 'backward')
        candidates{end + 1} = layer.backward; %#ok<AGROW>
    end
    if isfield(layer, 'block')
        block = layer.block;
        if isfield(block, 'forward')
            candidates{end + 1} = block.forward; %#ok<AGROW>
        end
    end

    for j = 1:numel(candidates)
        try
            info = functions(candidates{j});
        catch
            continue;
        end
        if ~isfield(info, 'workspace') || isempty(info.workspace)
            continue;
        end
        for k = 1:numel(info.workspace)
            ws = info.workspace{k};
            if isfield(ws, 'clsts') && isfield(ws, 'clstsAssign')
                clsts = ws.clsts;
                clstsAssign = ws.clstsAssign;
                fprintf('Found clsts/clstsAssign in layer %d\n', i);
                save(output_mat, 'clsts', 'clstsAssign', '-v7');
                return;
            end
        end
    end
end

error('Could not find clsts/clstsAssign in %s', input_mat);
end
