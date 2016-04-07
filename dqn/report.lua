-- plots
require 'hdf5'
require 'paths'

total_goals = 6

-- src = 'logs/golden'
-- src = 'logs/logs'
src = 'logs/final/'

runs = paths.dir(src)
print(runs)


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



local myFile = hdf5.open('stats.h5', 'w')


local function get_stats(dirs)
  local subgoal_hitrate = {} -- total of six subgoals
  local subgoal_chosen_meta = {}
  local subgoal_itr = {}
  for i=1,total_goals do
      subgoal_hitrate[i] = {}
      subgoal_chosen_meta[i] = {}
      subgoal_itr[i] = {}
  end
  for f in paths.files(dirs) do
      if string.match(f, "subgoal") then
          local itr = split(f,"_")[3]
          itr = split(itr,".t7")[1]
          local subg_stats = torch.load(src .. runs[expid] ..'/'.. f)
          local subg_success = subg_stats[1]
          local subg_total = subg_stats[2]
          for i =1,total_goals do 
              if subg_total[i+2] and subg_success[i+2] then --offset since subgoal id starts with 3
                  subgoal_hitrate[i][#subgoal_hitrate[i]+1] = subg_success[i+2]/subg_total[i+2]
                  subgoal_chosen_meta[i][#subgoal_chosen_meta[i]+1] = subg_total[i+2]
                  subgoal_itr[i][#subgoal_itr[i]+1] = itr
              end
          end
      end
  end

  for i=1,total_goals do 
    subgoal_itr[i] = torch.Tensor(subgoal_itr[i])
    subgoal_hitrate[i] = torch.Tensor(subgoal_hitrate[i])
    subgoal_chosen_meta[i] = torch.Tensor(subgoal_chosen_meta[i])
  end
  return subgoal_hitrate, subgoal_itr, subgoal_chosen_meta
end


for cnt = 1,#runs-2 do
  expid = cnt+2 --offset due to dir listing
  hirate = nil; itrs = nil
  hitrate, itrs, chosen_meta = get_stats('logs/latest/'..runs[expid])
  for i=1,total_goals do
    myFile:write('run' .. cnt .. '_subgoal_hitrate_gid_' .. i, hitrate[i])
    myFile:write('run' .. cnt .. '_subgoal_itr_gid_' .. i, itrs[i])
    myFile:write('run' .. cnt .. '_subgoal_total_gid_' .. i, chosen_meta[i])
  end
end

hitrate = nil
itrs = nil
hitrate, itrs,_ = get_stats('logs/golden/')
for i=1,total_goals do
  myFile:write('pretrain_subgoal_hitrate_gid_' .. i, hitrate[i])
  myFile:write('pretrain_subgoal_itr_gid_' .. i, itrs[i])
end

myFile:close()





 