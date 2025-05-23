# Progress report for the work on the DTM module and testing

## Log report

* 05/19 - G.A - First pytests presented 100% success, pycoverage presented 94% coverage
* 05/20 - G.A - Created tutorial notebook for DTM module as well as README, created report file for collaboration with Thea, added changes from original fork to Scott's repo under new branch 'dtm-module'
* 05/20 - G.A - Investigating lines 60, 101, 263-268 in __init__.py, responsible for the 4% not covered.
  * Line 60: def __call__(self, docs: Optional[list[list[str] | Doc]], labels: Optional[Iterable[str]]) -> None:
  * Line 101: def sorted_terms_list(self) -> list[str]:
  * Lines 263-268: Part of the to_df method's else block.
* 05/21 - G.A - Implemented new dedicated tests:
  * test_to_df_with_statistics_no_percentages(mock_df_dtm)
    * meant to cover for the to_df() method, as lines related to calculating "Total", "Mean", and "Median" statistics when the output is not in percentages (i.e., when as_percent=False) are being missed  
  * test_dtm_shape_property(mock_df_dtm)
    * dedicated test to call the shape property of a DTM instance.
* 05/21 - G.A - The __init__.py file's coverage has increased from 94% to 95%, and the number of missing statements has reduced from 6 to 5.
  * Missing column for __init__.py now lists 60, 263-268. Line 101 is no longer listed as missing
* 05/22 - G.A - Implemented new dedicated tests:
  * test_vectorizer_instantiation_and_call()
    * Verifies that the vectorizer can be instantiated and called with sample input.
  * test_to_df_handles_attribute_error_from_sparse(mock_df_dtm)
    * Ensures to_df() correctly handles AttributeError when the input is a sparse matrix.
  * test_to_df_handles_general_exception(mock_df_dtm)
    * Checks that to_df() gracefully handles unexpected exceptions during DataFrame conversion.
* 05/22 - G.A - __100% coverage achieved!__
* 05/22 - G.A - Formatted and linted with Ruff, added docstrings and typehinting to `test_dtm.py`
* 05/23 - G.A - Created `README for coding with Lexos`
* 05/23 - T.W - Finished writing `README.md` and `Tutorial for DTM module.ipynb` for dtm module

## To be done

* Improve existing tests
* Work on cleanup, making the code prettier and easier to read
  * Follow Google [guidelines](https://google.github.io/styleguide/pyguide.html)
* Implement newer tests for better coverage
* Work on tutorial notebook for dtm module
  * Consult from alpha version [tutorial](https://scottkleinman.github.io/lexos/api/dtm/)
* Work on README for dtm module
* Read on Tokenizer module [tutorial](https://scottkleinman.github.io/lexos/tutorial/tokenizing_texts/)
