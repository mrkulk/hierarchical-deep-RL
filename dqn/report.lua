-- plots
require 'hdf5'

src = 'logs/golden'
-- src = 'logs/logs'


function split(str, pat)
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
     table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end

subgoal_hitrate = {} -- total of six subgoals
subgoal_itr = {}
for i=1,6 do
    subgoal_hitrate[i] = {}
    subgoal_itr[i] = {}
end

for f in paths.files(src) do
    if string.match(f, "subgoal") then
        local itr = split(f,"_")[3]
        itr = split(itr,".t7")[1]
        local subg_stats = torch.load(src .. '/' .. f)
        local subg_success = subg_stats[1]
        local subg_total = subg_stats[2]
        for i =1,6 do 
            if subg_total[i+2] and subg_success[i+2] then --offset since subgoal id starts with 3
                subgoal_hitrate[i][#subgoal_hitrate[i]+1] = subg_success[i+2]/subg_total[i+2]
                subgoal_itr[i][#subgoal_itr[i]+1] = itr
            end
        end
    end
end

for i=1,6 do 
  subgoal_itr[i] = torch.Tensor(subgoal_itr[i])
  subgoal_hitrate[i] = torch.Tensor(subgoal_hitrate[i])
end

local myFile = hdf5.open('stats_basic.h5', 'w')
for i=1,6 do
  myFile:write('subgoal_hitrate_gid_' .. i, subgoal_hitrate[i])
  myFile:write('subgoal_itr_gid_' .. i, subgoal_itr[i])
end
myFile:close()





 