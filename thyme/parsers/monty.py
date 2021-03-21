"""
code copy from Pymatgen that use the Monty library
"""

import re
import numpy as np
from monty.io import zopen
from monty.re import regrep


def read_table_pattern(
    filename,
    header_pattern,
    row_pattern,
    footer_pattern,
    postprocess=str,
    last_one_only=True,
):
    r"""
    Parse table-like data. A table composes of three parts: header,
    main body, footer. All the data matches "row pattern" in the main body
    will be returned.

    Args:
        header_pattern (str): The regular expression pattern matches the
            table header. This pattern should match all the text
            immediately before the main body of the table. For multiple
            sections table match the text until the section of
            interest. MULTILINE and DOTALL options are enforced, as a
            result, the "." meta-character will also match "\n" in this
            section.
        row_pattern (str): The regular expression matches a single line in
            the table. Capture interested field using regular expression
            groups.
        footer_pattern (str): The regular expression matches the end of the
            table. E.g. a long dash line.
        postprocess (callable): A post processing function to convert all
            matches. Defaults to str, i.e., no change.
        last_one_only (bool): All the tables will be parsed, if this option
            is set to True, only the last table will be returned. The
            enclosing list will be removed. i.e. Only a single table will
            be returned. Default to be True.

    Returns:
        List of tables. 1) A table is a list of rows. 2) A row if either a list of
        attribute values in case the the capturing group is defined without name in
        row_pattern, or a dict in case that named capturing groups are defined by
        row_pattern.
    """

    with zopen(filename, "rt") as f:
        text = f.read()

    table_pattern_text = (
        header_pattern
        + r"\s*^(?P<table_body>(?:\s*"
        + row_pattern
        + r")+)\s+"
        + footer_pattern
    )
    table_pattern = re.compile(table_pattern_text, re.MULTILINE | re.DOTALL)
    rp = re.compile(row_pattern)

    tables = []
    for mt in table_pattern.finditer(text):
        table_body_text = mt.group("table_body")
        table_contents = []
        for line in table_body_text.split("\n"):
            ml = rp.search(line)
            # skip empty lines
            if not ml:
                continue
            d = ml.groupdict()
            if len(d) > 0:
                processed_line = {k: postprocess(v) for k, v in d.items()}
            else:
                processed_line = [postprocess(v) for v in ml.groups()]
            table_contents.append(processed_line)
        tables.append(table_contents)

    if last_one_only:
        retained_data = tables[-1]
    else:
        retained_data = tables
    return retained_data


def read_pattern(
    filename, patterns, reverse=False, terminate_on_match=False, postprocess=str
):
    r"""
    General pattern reading. Uses monty's regrep method. Takes the same
    arguments.

    Args:
        patterns (dict): A dict of patterns, e.g.,
            {"energy": r"energy\\(sigma->0\\)\\s+=\\s+([\\d\\-.]+)"}.
        reverse (bool): Read files in reverse. Defaults to false. Useful for
            large files, esp OUTCARs, especially when used with
            terminate_on_match.
        terminate_on_match (bool): Whether to terminate when there is at
            least one match in each key in pattern.
        postprocess (callable): A post processing function to convert all
            matches. Defaults to str, i.e., no change.

    Renders accessible:
        Any attribute in patterns. For example,
        {"energy": r"energy\\(sigma->0\\)\\s+=\\s+([\\d\\-.]+)"} will set the
        value of self.data["energy"] = [[-1234], [-3453], ...], to the
        results from regex and postprocess. Note that the returned values
        are lists of lists, because you can grep multiple items on one line.
    """
    matches = regrep(
        filename,
        patterns,
        reverse=reverse,
        terminate_on_match=terminate_on_match,
        postprocess=postprocess,
    )
    data = {}
    for k in patterns:
        data[k] = [i[0] for i in matches.get(k, [])]
    return data
