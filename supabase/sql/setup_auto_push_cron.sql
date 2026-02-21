-- Supabase scheduler for GitHub-triggered multi-user auto push
-- Replace GITHUB_PAT before running.

create extension if not exists pg_cron;
create extension if not exists pg_net;

-- Optional cleanup if you re-run this script
select cron.unschedule(jobid)
from cron.job
where jobname = 'auto_push_dispatch_each_minute';

select cron.schedule(
  'auto_push_dispatch_each_minute',
  '* * * * *',
  $$
  select net.http_post(
    url := 'https://api.github.com/repos/noblegasss/research-push/dispatches',
    headers := jsonb_build_object(
      'Accept', 'application/vnd.github+json',
      'Authorization', 'Bearer GITHUB_PAT',
      'X-GitHub-Api-Version', '2022-11-28',
      'Content-Type', 'application/json'
    ),
    body := jsonb_build_object('event_type', 'supabase-auto-push')
  );
  $$
);

-- Verify jobs
select jobid, jobname, schedule, command
from cron.job
where jobname = 'auto_push_dispatch_each_minute';
