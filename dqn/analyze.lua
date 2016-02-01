-- analyze subgoal success

expname = 'eps_endt=200000_lr=0.00025_port=9000_usedist=true'
dir = 'logs/' .. expname

finalstats = {}
for i=3,8 do
	finalstats[i] = 0
end

for f in paths.files(dir) do
	if string.match(f, "subgoal") then
		local stats = torch.load(dir .. '/' .. f)
		-- print(stats[1])
		for sid, hitrate in pairs(stats[1]) do
			finalstats[sid] = finalstats[sid] + stats[1][sid]
		end
	end
end

print('Final subgoal stats {key=subgoal-id, value=hitrate:')
print(finalstats)
