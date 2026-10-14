// Date-equality assertion example.
predicate MatchesRequestedDate(requested: int, proposed: int)
{
  requested == proposed
}

method CheckProposedDate()
{
  assert MatchesRequestedDate(20261110, 20261110);
}
