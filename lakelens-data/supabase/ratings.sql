-- Visitor ratings for lakelens.html (Report tab). Run once in Supabase: SQL Editor → New query → Run.
-- The page uses the project's public anon/publishable key, so row-level security is what keeps this safe:
-- anyone can add a rating and read the last day of ratings; nobody can edit or delete through the API.

create table if not exists public.ratings (
  id          bigint generated always as identity primary key,
  lake        text not null check (lake in ('lew', 'grp', 'rr', 'tex')),
  crowding    smallint check (crowding between 0 and 5),
  roughness   smallint check (roughness between 0 and 5),
  cleanliness smallint check (cleanliness between 0 and 5),
  created_at  timestamptz not null default now(),
  check (crowding is not null or roughness is not null or cleanliness is not null)
);

create index if not exists ratings_recent on public.ratings (created_at desc);

alter table public.ratings enable row level security;

grant select, insert on public.ratings to anon;

-- created_at must be "now", so nobody can backdate or pre-date ratings
create policy "anyone can add a rating" on public.ratings
  for insert to anon
  with check (created_at between now() - interval '1 minute' and now() + interval '1 minute');

create policy "anyone can read the last day" on public.ratings
  for select to anon
  using (created_at > now() - interval '1 day');

-- Cleanup: the page only uses the last 2 hours and the public key can't read past a day,
-- so delete anything older than a day. Runs at the top of every hour (pg_cron, built into Supabase).
-- Re-running this replaces the job of the same name. Check it with: select * from cron.job;
create extension if not exists pg_cron;

select cron.schedule(
  'delete-old-ratings',
  '0 * * * *',
  $$delete from public.ratings where created_at < now() - interval '1 day'$$
);
