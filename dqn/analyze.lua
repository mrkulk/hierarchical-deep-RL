-- analyze subgoal success

expname = 'basic2'
dir = 'logs/' .. expname

finalstats = {}
finalcount = {}
for i=3,8 do
	finalstats[i] = 0
	finalcount[i] = 0
end

for f in paths.files(dir) do
	if string.match(f, "subgoal") then
		local stats = torch.load(dir .. '/' .. f)
		-- print(stats[1])
		for sid, hitrate in pairs(stats[1]) do
			finalstats[sid] = finalstats[sid] + stats[1][sid]
			finalcount[sid] = finalcount[sid] + stats[2][sid]
		end
	end
end

for i=3,8 do
	if finalstats[i] > 0 then
		finalstats[i] = finalstats[i]/finalcount[i]
	end
end

print('Final subgoal stats {key=subgoal-id, value=% hitrate:')
print(finalstats)
