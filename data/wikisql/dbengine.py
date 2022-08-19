import sqlite3
import re
from babel.numbers import parse_decimal, NumberFormatError


schema_re = re.compile(r'\((.+)\)') # group (.......) dfdf (.... )group
num_re = re.compile(r'[-+]?\d*\.\d+|\d+') # ? zero or one time appear of preceding character, * zero or several time appear of preceding character.
# Catch something like -34.34, .4543,
# | is 'or'

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']

class DBEngine:

    def __init__(self, fdb):
        #fdb = 'data/test.db'
        #self.db = records.Database('sqlite:///{}'.format(fdb))
        self.conn = sqlite3.connect(fdb)

    def query(self, data):
        return self.execute(data['table_id'], data['sql']['sel'], data['sql']['agg'], data['sql']['conds'])
    
    def execute_query(self, table_id, query, json=True):
        if json:
            return self.execute(table_id, query['sql']['sel'], query['sql']['agg'], query['sql']['conds'])
        else:
            return self.execute(table_id, query.sel_index, query.agg_index, query.conditions)
    

    def execute(self, table_id, select_index, aggregation_index, conditions, lower=True):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        table_info = self.conn.execute("SELECT sql from sqlite_master WHERE tbl_name = '{}'".format(table_id)).fetchall()[0]
        schema_str = schema_re.findall(table_info[0])[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        select = 'col{}'.format(select_index)
        agg = agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and (isinstance(val, str) or isinstance(val, str)):
                val = val.lower()
            if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):
                try:
                    val = float(parse_decimal(val, locale='en_US'))
                except NumberFormatError as e:
                    try:
                        val = float(num_re.findall(val)[0]) # need to understand and debug this part.
                    except:
                        # Although column is of number, selected one is not number. Do nothing in this case.
                        pass
            where_clause.append("col{} {} '{}'".format(col_index, cond_ops[op], str(val).replace("'","''").replace("\\","\\\\")))
            where_map['col{}'.format(col_index)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        
        query = 'SELECT {} AS result FROM {} {}'.format(select, table_id, where_str)
        out = self.conn.execute(query)
        return out.fetchall()

    def show_table(self, table_id):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        rows = self.conn.execute('select * from ' +table_id).fetchall()
        print(rows)
        
    
    def get_item_index(self, item_list, item, tokenizer):
        item = item.lower()
        for i, h in enumerate(item_list):
            if item == h.lower() or tokenizer.encode(item)==tokenizer.encode(h.lower()):
                return i
        return -1

    def get_where_value(self, tokenizer, orig_question, where_value):
        lower_question = orig_question.lower()
        idx = lower_question.find(where_value)
        idx_orig = orig_question.find(where_value)
        if idx > -1 and idx_orig == -1:
            return (idx, orig_question[idx:idx+len(where_value)].replace('\'','"'))
        elif idx_orig > -1:
            return (idx_orig, where_value.replace('\'','"'))
        else:
            return (-1, where_value.replace('\'','"'))

    def generate_logical_form(self, tokenizer, sql_string, orig_question, table, agg_ops, cond_ops, keyname, debug = False):
        sql_string = sql_string.lower().strip()
        logical_form = {}
        from_pos_idx = sql_string.find(' from table')
        where_pos_idx = sql_string.find(' where [')
        if debug:
            print(sql_string, "from pos:", from_pos_idx, sql_string[from_pos_idx:from_pos_idx+5], 
                    "where pos", where_pos_idx, sql_string[where_pos_idx:where_pos_idx+6])

        # extract select columns and aggregation function
        select_string = sql_string[7:from_pos_idx].strip()
        if debug:
            print("select_string", select_string)

        if select_string[0] != '[':
            # aggregation function
            sep = select_string.find("(") 
            logical_form['sel'] = self.get_item_index(table['header'], select_string[sep+2:-2], tokenizer)
            logical_form['agg'] = self.get_item_index(agg_ops, select_string[:sep], tokenizer)
        else:
            logical_form['sel'] = self.get_item_index(table['header'], select_string[1:-1], tokenizer)
            logical_form['agg'] = 0

        if debug:
            print("logical form so far: ", logical_form)

        # extract where conditions
        logical_form['conds'] =[]
        logical_form['where_value_idx'] = []
        if where_pos_idx > 0:
            where_string = sql_string[where_pos_idx+7:].strip()
            if debug:
                print("where_string:", where_string)
            conds = where_string.split(' and [')
            for i, c in enumerate(conds):
                closing_bracket = c.find('] ')
                if i == 0:
                    where_column = c[1:closing_bracket]
                else:
                    where_column = c[:closing_bracket]

                where_op = c[closing_bracket+2:closing_bracket+3]

                if debug:
                    print("where value:", c[closing_bracket+4:])
                
                if c[closing_bracket+3] == ' ':
                    if c[closing_bracket+4] == "'":
                        where_value = c[closing_bracket+5:-1]
                    else:
                        where_value = c[closing_bracket+4:]
                else:
                    if c[closing_bracket+3] == "'":
                        where_value = c[closing_bracket+4:-1]
                    else:
                        where_value = c[closing_bracket+3:]
                
                if debug:
                    print("Getting where value", where_value)
                
                where_idx, where_val = self.get_where_value(tokenizer, orig_question, where_value)
                
                if debug:
                    print("where column:", where_column, " where op:", where_op, " where value:", 
                            where_idx, where_val)
                logical_form['conds'].append([self.get_item_index(table['header'],where_column, tokenizer),
                                                self.get_item_index(cond_ops, where_op, tokenizer),
                                                where_val])
                logical_form['where_value_idx'].append([where_idx])
        if debug:
            print(logical_form)
        result = {keyname:logical_form, "table_id": table['id']}
        return result