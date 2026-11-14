create table if not exists searches (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  kind text not null,
  source text not null,
  compare_mode text not null,
  surface text not null,
  body_a text,
  body_b text,
  lat double precision,
  lon double precision,
  label text,
  result_count integer not null default 0,
  summary text not null
);

create index if not exists searches_created_at_idx on searches (created_at desc);
