# Output format

Single question (`grades_question_8.json`):

```json
{
  "question": 8,
  "total_students": 50,
  "conflicts": 3,
  "grades": [
    {
      "student_id": "400000003",
      "question": 8,
      "final_grade": 85,
      "conflict": false,
      "model_1": {"name": "openai/gpt-4.1-nano", "grade": 85, "reasoning": "..."},
      "model_2": {"name": "google/gemini-2.5-flash", "grade": 82, "reasoning": "..."}
    }
  ]
}
```

All questions (`grades_result.json`):

```json
{
  "num_questions": 10,
  "total_students": 50,
  "students_with_conflicts": 3,
  "grades": [
    {
      "student_id": "400000003",
      "final_grades": [85, 70, 100, null, 90],
      "total_grade": 345,
      "conflicts": [false, false, false, true, false],
      "has_conflict": true,
      "model_1": {"name": "openai/gpt-4.1-nano", "grades": [85, 70, 100, 30, 90], "reasoning": "..."},
      "model_2": {"name": "google/gemini-2.5-flash", "grades": [85, 70, 100, null, 90], "reasoning": "..."}
    }
  ]
}
```

`final_grade`/`final_grades` entries are `null` when neither model produced a
grade. `total_grade` sums only the grades that exist.

