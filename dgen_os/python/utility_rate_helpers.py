import ast

def _unwrap_singleton(x):
    """
    If x is like array([<one thing>]) / [<one thing>] / ( <one thing>, ),
    return the single element; else return x unchanged.
    """
    if hasattr(x, "shape") and hasattr(x, "tolist"):
        try:
            lst = x.tolist()
            if isinstance(lst, list) and len(lst) == 1:
                return lst[0]
        except Exception:
            pass

    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]

    return x


def update_pysam_tariff_dict_cell_all_rows(
    tariff_cell,
    *,
    new_monthly_fixed_charge=None,
    new_volumetric_rate=None,
    tou_mat_rate_index=4,
    return_as="string",
):
    """
    Update ONE df cell that contains a PySAM-ready tariff dict.

    Updates:
      - ur_monthly_fixed_charge (if provided)
      - ur_ec_tou_mat volumetric rate for *every* row (all tiers & TOU periods) (if provided)

    Parameters
    ----------
    tariff_cell : Any
        The value from df["tariff_dict"]. Can be dict, stringified dict, or singleton array/list containing that.
    new_monthly_fixed_charge : float | None
    new_volumetric_rate : float | None
    tou_mat_rate_index : int
        Index in each ur_ec_tou_mat tuple corresponding to the $/kWh rate. In your examples it's 4.
    return_as : {"string","dict"}

    Returns
    -------
    Updated tariff as stringified dict (default) or dict.
    """
    raw = _unwrap_singleton(tariff_cell)

    if isinstance(raw, dict):
        tariff = raw
    elif isinstance(raw, str):
        tariff = ast.literal_eval(raw)
        if not isinstance(tariff, dict):
            raise ValueError("Parsed tariff_cell string but did not get a dict.")
    else:
        raise TypeError(f"Unsupported tariff_cell type: {type(raw)}")

    if new_monthly_fixed_charge is not None:
        tariff["ur_monthly_fixed_charge"] = float(new_monthly_fixed_charge)

    if new_volumetric_rate is not None:
        if "ur_ec_tou_mat" not in tariff or tariff["ur_ec_tou_mat"] is None:
            raise KeyError("tariff dict has no 'ur_ec_tou_mat' to update.")
        new_rate = float(new_volumetric_rate)

        updated = []
        for row in tariff["ur_ec_tou_mat"]:
            if not isinstance(row, (list, tuple)):
                raise TypeError(f"ur_ec_tou_mat row is not tuple/list: {row!r}")
            row_list = list(row)
            if tou_mat_rate_index >= len(row_list):
                raise IndexError(
                    f"tou_mat_rate_index={tou_mat_rate_index} out of range for row len={len(row_list)}: {row!r}"
                )
            row_list[tou_mat_rate_index] = new_rate
            updated.append(tuple(row_list))

        tariff["ur_ec_tou_mat"] = updated

    if return_as == "dict":
        return tariff
    if return_as == "string":
        return str(tariff)
    raise ValueError("return_as must be 'string' or 'dict'.")


def apply_tariff_updates(
    df,
    *,
    tariff_col="tariff_dict",
    eia_col="eia_id",
    tariff_name_col="tariff_name",
    eia_id=None,
    tariff_name=None,
    new_monthly_fixed_charge=None,
    new_volumetric_rate=None,
    tou_mat_rate_index=4,
):
    """
    Apply tariff updates across df, optionally filtering by eia_id and/or tariff_name.

    - If eia_id is provided, only rows with that eia_id are updated.
    - If tariff_name is provided, only rows with that tariff_name are updated.
    - If both are provided, both filters apply (intersection).

    Returns a COPY of df with updated tariff_col.
    """
    out = df.copy()

    mask = True
    if eia_id is not None:
        mask = mask & (out[eia_col].astype(str) == str(eia_id))
    if tariff_name is not None:
        mask = mask & (out[tariff_name_col].astype(str) == str(tariff_name))

    if mask is True:
        # no filtering requested; update all rows
        idx = out.index
    else:
        idx = out.index[mask]

    out.loc[idx, tariff_col] = out.loc[idx, tariff_col].apply(
        lambda cell: update_pysam_tariff_dict_cell_all_rows(
            cell,
            new_monthly_fixed_charge=new_monthly_fixed_charge,
            new_volumetric_rate=new_volumetric_rate,
            tou_mat_rate_index=tou_mat_rate_index,
            return_as="string",
        )
    )
    return out