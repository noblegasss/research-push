select cron.unschedule(jobid)
from cron.job
where jobname = 'auto_push_dispatch_each_minute';
