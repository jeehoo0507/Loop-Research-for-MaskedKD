
## round 1 (20261030_135959, researcher claude): locked files were modified and reverted to 0671ec21450574539b6bb2f2a63a31adec9532f7
Jobs submitted in this round need re-checking.

## round 55 (20261031_081027, researcher claude): locked files were modified and reverted to 1dea35ebe47c722550dec69c664febc62fde75d7
Jobs submitted in this round need re-checking.

## note on round 55 (20261031_081027): not tampering
The "modified" locked files were the human replan (commit eb0e3ff: CUB search, mean accuracy + FLOPs, CE lr sweep),
pulled into the tree mid-round; the tamper check reverted it in c5753fd. The replan is restored and the researcher can
no longer pull or push. Jobs from that round are not suspect on this account.
