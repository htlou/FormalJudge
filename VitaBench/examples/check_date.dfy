// Date-equality assertion example.
// Positive assertion for the mismatching date: expected to remain unproven.
predicate MatchesRequestedDate(requested: int, proposed: int)
{
  requested == proposed
}

method CheckProposedDate()
{
  assert MatchesRequestedDate(20261110, 20261111);
}
