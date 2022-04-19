-- CTCI Chapter 14 - SQL solutions


-- ############################################################################################
-- 14.1 Multiple Apartments -- Write a SQL query to get a list of tenants who are renting more than one apartment.

SELECT t.TenantID, t.TenantName, COUNT(a.AptID) AS num_apartments_rented
FROM
  Tenants t
  LEFT JOIN AptTenants a ON a.TenantID = t.TenantID
GROUP BY t.TenantID, t.TenantName
HAVING COUNT(a.AptID) > 1


-- ############################################################################################
-- 14.2 Open Requests: Write a SQL query to get a list of all buildings and the number of open requests (Requests in which status equals 'Open').

SELECT b.BuildingID, SUM(CASE WHEN r.Status = 'Open' THEN 1 ELSE 0 END) AS num_open_requests
FROM 
  Buildings b
  LEFT JOIN Apartments a ON a.BuildingID = b.BuildingID
  LEFT JOIN Requests r ON r.AptID = a.AptID 
GROUP BY b.BuildingID


-- ############################################################################################
-- 14.3 Close All Requests: Building #11 is undergoing a major renovation. Implement a query to close all requests from apartments in this building.

UPDATE Requests
SET 
  status = 'Closed'
FROM 
  Requests r 
  LEFT JOIN Apartments a ON a.AptID = r.AptID 
  -- LEFT JOIN Buildings b on b.BuildingID = a.BuildingID -- Not needed!
WHERE 
  a.BuildingID = 11
  -- Or, without JOIN above: WHERE AptID IN (SELECT AptID FROM Apartments WHERE BuildingID = 11)


-- ############################################################################################
-- 14.4 Joins: What are the different types of joins? Please explain how they differ and why certain types are better in certain situations.

/*
  JOINs combine the results of two tables.

  CROSS JOIN: Cartesian product -- all rows from A are joined with all rows from B. Final row count = |A| * |B|
  
  FULL OUTER JOIN: Join all elements, but only retain as many rows as needed, depending on the joining logic. I.e. If we join on ID, then if a certain ID is present in A but not in B, then we will still list those rows of A, and vice versa -- whereas if it is present in both, then we will merge rows when we can (depending on joining logic).
  
  LEFT JOIN: Only keep rows that have a matching JOIN (from LEFT table). RIGHT JOIN the same but for RIGHT table.
  
  INNER JOIN: Only keep rows that have a matching JOIN from both tables.  
*/


-- ############################################################################################
-- 14.5 Denormalization: What is denormalization? Explain the pros and cons. 
/* 
  Denormalization is a database optimzation technique that aims to optimize for read efficiency. I.e. speed up database reads by duplicating information across tables.
  
  The opposite of this technique is database normalization, where we aim to minimize redundancy by avoiding duplicating data, and instead providing logical relations between tables through primary and foreign keys. This minimizes the costs of updating existing data, and uses up less storage space, but requires expensive JOINs to have a full view of the data.
  
  Pros/Cons of Denormalization:
    Pros: Faster database reads, because we don't need to JOIN as often (or at all). Queries are easier to implement, maintain and debug.
    Cons: UPDATES and INSERTS are more expensive and more complicated to implement. Data consistency is harder to guarantee (which version of the same piece of data is the source of truth in case there is a discrepancy?). More storage used up due to duplicate data.
    
  When operating at scale, we can have different requirements (time SLAs, storage limitations, processing limitations, etc.) depending on the product/department. Both normalization and denormalization techniques can be relevant depending on these requirements.
*/


-- ############################################################################################
-- 14.6 Entity-Relationship Diagram: Draw an entity-relationship diagram for a database with companies, people, and professionals (people who work for companies).
/*
  Normalized (Professionals links People to Companies):
    Companies: CompanyID pk, CompanyName, etc.
    People: PeopleID pk, PeopleName, etc.
    Professionals: PeopleID fk, CompanyID fk

  Denormalized:
    Companies: CompanyID pk, CompanyName, etc. + PeopleID fk  // Several rows per Company, for each Person working there 
    People: PeopleID pk, PeopleName, etc. + CompanyID fk  // Several rows per Person, for each Company they work at
    
  Considerations:
    If a Person is guaranteed to only work at 0 or 1 companies, then we can include CompanyID fk into People, and would not need Professionals as an extra table in the Normalized design.
*/


-- ############################################################################################
-- 14.7 Design Grade Database: Imagine a simple database storing information for students' grades. Design what this database might look like and provide a SQL query to return a list of the honor roll students (top 10%), sorted by their grade point average.
/*
  Students:
    StudentID pk,
    StudentName
    
  Grades:
    GradeID pk,
    StudentID fk,
    Grade
*/

SELECT 
    StudentID
  , StudentName
  , Grade
  , percentile
FROM (
  SELECT 
      s.StudentID
    , s.StudentName
    , g.Grade
    , percent_rank() OVER(ORDER BY g.Grade DESC) AS percentile_rank
    , cume_dist() OVER(ORDER BY g.Grade DESC) AS cume_dist_rank
  FROM
    Students s
    LEFT JOIN Grades g ON g.StudentID = s.StudentID 
) ranked
WHERE percentile_rank <= 0.1 OR cume_dist_rank <= 0.1  -- by using both, we ensure we include students even if they go beyond the naive percentile? -- TODO: Test if true?
ORDER BY g.Grade DESC, s.StudentName ASC, s.StudentID ASC

-- Alternatively, establish the smallest Grade in the naive top 10% in a subquery, then SELECT all students with at least that Grade using an outer query.
WITH Student_Grades AS (
    SELECT 
        s.StudentID
      , s.StudentName
      , g.Grade 
      , percent_rank() OVER(ORDER BY g.Grade DESC) AS percentile_rank
    FROM 
      Students s
      LEFT JOIN Grades g ON g.StudentID = s.StudentID 
)
SELECT 
    StudentID
  , StudentName
  , Grade
  , percentile
FROM Student_Grades
WHERE
  Grade >= (
    SELECT MIN(Grade) FROM Student_Grades WHERE percentile_rank <= 0.1
  )
ORDER BY Grade DESC, StudentName ASC, StudentID ASC
