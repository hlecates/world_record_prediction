# NESCAC Swimming Data Project

## Why Manual Data Entry is Needed

Although the parsing pipeline has decent performance—successfully parsing approximately 24 entries for every finals event—there are still significant issues due to the irregularity of recorded data across formats and the nature of swimming results themselves. Because the dataset consists of only one event per year, the sample size is small and it is not feasible to simply drop problematic events or tolerate mutated data. 

To address this, the `manual_update` pipeline was developed. This workflow allows the user to:
- Convert parsed CSV files into human-readable text files for easy manual review and correction.
- Edit and clean up event and swimmer data directly, ensuring accuracy and consistency.
- Convert the corrected text files back into CSV format, overwriting the originals and maintaining a clean dataset.

This approach ensures that all events are preserved and the data quality is as high as possible, even when automated parsing fails or produces errors due to inconsistent source formatting.

Some name were cutoff in the prelims such as kearns --> kear in the prelims in 2002, to many to simply hardcode, and

---

https://nescac.com/sports/2020/7/14/championships-pastchamps-msd.aspx


should be cleaner data, need to try multiple parsing techniques for each results

some have results in tables on html
some have two col pdfs
and others have 


Had to drop 2002, 2003, 2004, 2008, since no seed time were included, hence no featured would be able to be created for the targets