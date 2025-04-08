CREATE TABLE IF NOT EXISTS ds_qa (
    id SERIAL PRIMARY KEY,
    section TEXT,
    subsection TEXT,
    question TEXT,
    answer TEXT,
    hash_answer TEXT,
    UNIQUE (section, subsection, question)
);