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


-- 學生學號
CREATE TABLE student (
    student_id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(50)
);

INSERT INTO student (student_id, name) VALUES
('S001', 'Alice'),
('S002', 'Bob'),
('S003', 'Charlie'),
('S004', 'Mimi'),
('S005', 'Dora'),
('S006', 'Cindy'),
('S007', 'Patty'),
('S008', 'Vicky');




-- 插入 login_log 資料
INSERT INTO login_log (student_id, login_time) VALUES
('S001', '2025-05-01 08:00:00'),
('S002', '2025-05-01 08:15:00'),
('S003', '2025-05-01 08:30:00'),
('S001', '2025-05-02 08:05:00'),
('S004', '2025-05-02 08:10:00'),
('S005', '2025-05-02 08:20:00'),
('S002', '2025-05-03 08:25:00'),
('S006', '2025-05-03 08:30:00'),
('S007', '2025-05-03 08:35:00'),
('S003', '2025-05-04 08:40:00');

-- 插入 homework_submission 資料
INSERT INTO homework_submission (student_id, homework_id, submitted) VALUES
('S001', 1, 1),
('S002', 1, 1),
('S003', 1, 0),
('S004', 1, 1),
('S005', 1, 1),
('S001', 2, 1),
('S002', 2, 0),
('S003', 2, 1),
('S006', 1, 1),
('S007', 1, 0);

-- 插入 exam_scores 資料
INSERT INTO exam_scores (student_id, exam_id, score) VALUES
('S001', 1, 88.5),
('S002', 1, 92.0),
('S003', 1, 76.0),
('S004', 1, 85.5),
('S005', 1, 90.0),
('S001', 2, 79.0),
('S002', 2, 84.5),
('S003', 2, 80.0),
('S006', 1, 70.0),
('S007', 1, 65.5);




