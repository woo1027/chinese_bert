-- 登入紀錄
CREATE TABLE login_log (
    student_id VARCHAR(10),
    login_time DATETIME
);

-- 作業上傳
CREATE TABLE homework_submission (
    student_id VARCHAR(10),
    homework_id INT,
    submitted BIT
);

-- 考試成績
CREATE TABLE exam_scores (
    student_id VARCHAR(10),
    exam_id INT,
    score FLOAT
);


