-- analyze subgoal success

--expname = 'eps_endt=200000_lr=0.00025_port=9000_usedist=true'
--expname = 'meta10'
--expname = 'meta15_all_priority'
--expname = 'meta14_all_death50'
expname = 'meta16_priority_noclone'
--expname = 'meta13_all'
--expname = 'basic2'
--

cmd = torch.CmdLine()
cmd:option('-expname','meta10','directory of logs')

params = cmd:parse(arg)

dir = params.expname

print("Stats from file:", params.expname)

finalstats = {}
finalcount = {}
for i=3,8 do
	finalstats[i] = 0
	finalcount[i] = 0
end

for f in paths.files(dir) do
    if string.match(f, "subgoal_statistics") then
        -- print(f)
    	local stats = torch.load(dir .. '/' .. f)
    	--print('--------')
    	-- print(stats[1])
    	--print(stats[3])
    	for sid, hitrate in pairs(stats[2]) do
            if stats[1][sid] then
    		    finalstats[sid] = finalstats[sid] + stats[1][sid]
            end
    		finalcount[sid] = finalcount[sid] + stats[2][sid]
    	end
    
    	--for i=3,8 do
    	--	if stats[2][i] and stats[2][i] > 0  then
    	--		top = stats[1][i]  or 0
    	--		print(i, ":", top/stats[2][i], stats[2][i])
    	--	end
    	--end    
    end
end

for i=3,8 do
	if finalstats[i] > 0 then
		finalstats[i] = finalstats[i]/finalcount[i]
	end
end

print('Final subgoal stats {key=subgoal-id, value=% hitrate:')
print(finalstats)
print(finalcount)
